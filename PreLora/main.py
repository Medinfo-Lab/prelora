from tools.PreLora import create_prelora_model, count_trainable_parameters, get_trainable_parameters
import torch
if __name__ == '__main__':
    # 创建PreLora模型
    model = create_prelora_model(
        model_name="bert-base-chinese",
        num_labels=21,  # ICD-10类别数
        prefix_length=10,
        lora_rank=8,
        prefix_hidden_size=512
    )

    # 检查可训练参数
    trainable_params = count_trainable_parameters(model)
    print(f"可训练参数数量: {trainable_params}")

    # 训练模型
    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=1e-3)