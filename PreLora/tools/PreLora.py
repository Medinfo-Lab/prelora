import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional, Dict, Any


class PrefixEncoder(nn.Module):
    """
    MLP编码器用于生成可学习的连续前缀提示
    """

    def __init__(self, config, prefix_hidden_size: int = 512):
        super().__init__()
        self.prefix_length = config.prefix_length
        self.hidden_size = config.hidden_size
        self.prefix_hidden_size = prefix_hidden_size

        # MLP编码器
        self.embedding = nn.Embedding(self.prefix_length, self.hidden_size)
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.prefix_hidden_size),
            nn.Tanh(),
            nn.Linear(self.prefix_hidden_size, config.num_hidden_layers * 2 * self.hidden_size)
        )

    def forward(self, prefix_ids: torch.Tensor):
        # prefix_ids: [prefix_length]
        prefix_embeds = self.embedding(prefix_ids)  # [prefix_length, hidden_size]
        past_key_values = self.transform(prefix_embeds)  # [prefix_length, num_layers * 2 * hidden_size]

        # 重塑为 [num_layers, 2, prefix_length, hidden_size]
        past_key_values = past_key_values.view(
            self.prefix_length, -1, 2, self.hidden_size
        )
        past_key_values = past_key_values.permute(1, 2, 0, 3)  # [num_layers, 2, prefix_length, hidden_size]

        return past_key_values


class LoRALayer(nn.Module):
    """
    LoRA低秩适配层
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA矩阵A和B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, in_features] or [batch_size, in_features]
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return lora_output


class LoRALinear(nn.Module):
    """
    带有LoRA适配的线性层
    """

    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )

    def forward(self, x: torch.Tensor):
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        return linear_output + lora_output


class PreLoraAttention(nn.Module):
    """
    集成Prefix Tuning和LoRA的注意力机制
    """

    def __init__(self, original_attention, prefix_length: int = 10, lora_rank: int = 8):
        super().__init__()
        self.original_attention = original_attention
        self.prefix_length = prefix_length
        self.hidden_size = original_attention.hidden_size

        # 用LoRALinear替换原始线性层
        self.query = LoRALinear(original_attention.q_proj, lora_rank)
        self.key = LoRALinear(original_attention.k_proj, lora_rank)
        self.value = LoRALinear(original_attention.v_proj, lora_rank)
        self.dense = LoRALinear(original_attention.o_proj, lora_rank)

    def forward(self, hidden_states: torch.Tensor, past_key_values: Optional[torch.Tensor] = None, **kwargs):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape

        # 计算Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # 如果有前缀key-values，则拼接
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
            prefix_seq_len = past_key.shape[1]
        else:
            prefix_seq_len = 0

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.hidden_size ** 0.5)

        # 注意力掩码
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            if prefix_seq_len > 0:
                # 为前缀部分扩展注意力掩码
                prefix_mask = torch.ones(
                    batch_size, 1, seq_len, prefix_seq_len,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            attention_scores = attention_scores + attention_mask

        # 应用softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 注意力输出
        context_layer = torch.matmul(attention_probs, value)
        attention_output = self.dense(context_layer)

        return attention_output


class PreLoraModel(nn.Module):
    """
    PreLora主模型：结合Prefix Tuning和LoRA
    """

    def __init__(self, base_model: PreTrainedModel, prefix_length: int = 10,
                 lora_rank: int = 8, prefix_hidden_size: int = 512):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.prefix_length = prefix_length
        self.lora_rank = lora_rank

        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 前缀编码器
        self.prefix_encoder = PrefixEncoder(
            self.config,
            prefix_hidden_size=prefix_hidden_size
        )

        # 前缀ID
        self.prefix_ids = torch.arange(prefix_length).long()

        # 用PreLoraAttention替换原始注意力层
        self._replace_attention_layers()

    def _replace_attention_layers(self):
        """替换模型中的注意力层为PreLoraAttention"""
        if hasattr(self.base_model, 'transformer'):
            # 对于GPT-like模型
            for layer in self.base_model.transformer.h:
                layer.attn = PreLoraAttention(
                    layer.attn,
                    self.prefix_length,
                    self.lora_rank
                )
        elif hasattr(self.base_model, 'encoder'):
            # 对于BERT-like模型
            for layer in self.base_model.encoder.layer:
                layer.attention.self = PreLoraAttention(
                    layer.attention.self,
                    self.prefix_length,
                    self.lora_rank
                )
        else:
            raise ValueError("Unsupported model architecture")

    def get_prefix_embeddings(self):
        """获取前缀嵌入"""
        device = next(self.parameters()).device
        prefix_ids = self.prefix_ids.to(device)
        return self.prefix_encoder(prefix_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        # 获取前缀key-values
        past_key_values = self.get_prefix_embeddings()

        # 准备模型输入
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            **kwargs
        }

        # 前向传播
        outputs = self.base_model(**model_inputs)

        return outputs


class PreLoraForSequenceClassification(nn.Module):
    """
    用于序列分类的PreLora模型
    """

    def __init__(self, base_model: PreTrainedModel, num_labels: int,
                 prefix_length: int = 10, lora_rank: int = 8,
                 prefix_hidden_size: int = 512):
        super().__init__()
        self.prelora_model = PreLoraModel(
            base_model, prefix_length, lora_rank, prefix_hidden_size
        )
        self.num_labels = num_labels

        # 分类头
        self.classifier = nn.Linear(
            base_model.config.hidden_size,
            num_labels
        )

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs):

        # 获取模型输出
        outputs = self.prelora_model(input_ids, attention_mask, **kwargs)

        # 获取序列表示（取第一个token或平均池化）
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]

        # 池化策略：取第一个token [CLS]
        pooled_output = sequence_output[:, 0, :]

        # 分类
        logits = self.classifier(pooled_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }


def get_trainable_parameters(model: nn.Module) -> list:
    """获取模型中需要训练的参数"""
    return [param for param in model.parameters() if param.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    """计算可训练参数数量"""
    return sum(p.numel() for p in get_trainable_parameters(model))


# 使用示例
def create_prelora_model(model_name: str, num_labels: int, **kwargs):
    """
    创建PreLora模型

    Args:
        model_name: 基础模型名称
        num_labels: 分类标签数量
        **kwargs: PreLora参数

    Returns:
        PreLora模型实例
    """
    from transformers import AutoModel, AutoConfig

    # 加载基础模型
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # 创建PreLora模型
    prelora_model = PreLoraForSequenceClassification(
        base_model=base_model,
        num_labels=num_labels,
        **kwargs
    )

    return prelora_model


# 训练循环示例
def train_prelora(model, train_dataloader, optimizer, device):
    """训练PreLora模型"""
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dataloader)