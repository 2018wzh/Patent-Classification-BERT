import json
import os
import subprocess

CONFIG_FILE = "config/train_config.json"

def load_config():
    """加载配置文件"""
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件 {CONFIG_FILE} 不存在。")
        return None
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    """保存配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"配置已保存到 {CONFIG_FILE}")

def get_input(prompt, current_value):
    """获取用户输入，并处理类型转换"""
    user_input = input(f"{prompt} (当前: {current_value}, 直接回车跳过): ").strip()
    if not user_input:
        return current_value

    # 尝试转换为原始值的类型
    original_type = type(current_value)
    if original_type == bool:
        if user_input.lower() in ['true', 't', 'yes', 'y', '1']:
            return True
        elif user_input.lower() in ['false', 'f', 'no', 'n', '0']:
            return False
        else:
            print("无效的布尔值，将保留原值。")
            return current_value
    elif original_type == int:
        try:
            return int(user_input)
        except ValueError:
            print("无效的整数，将保留原值。")
            return current_value
    elif original_type == float:
        try:
            return float(user_input)
        except ValueError:
            print("无效的浮点数，将保留原值。")
            return current_value
    elif current_value is None: # 处理gpus这种初始为None的情况
        return user_input if user_input else None
    else:
        return user_input

def interactive_configure():
    """交互式配置训练参数"""
    config = load_config()
    if config is None:
        return

    print("欢迎使用训练配置向导。对于每个选项，输入新值或直接按回车键保留当前值。")
    print("-" * 30)

    new_config = {}
    for key, value in config.items():
        new_value = get_input(f"设置 '{key}'", value)
        new_config[key] = new_value

    print("-" * 30)
    print("新配置如下:")
    print(json.dumps(new_config, indent=4, ensure_ascii=False))
    
    confirm = input("是否保存新配置? (y/n): ").strip().lower()
    if confirm == 'y':
        save_config(new_config)
    else:
        print("配置未保存。")

def run_training():
    """使用当前配置启动训练"""
    config = load_config()
    if config is None:
        return
        
    print("\n即将使用以下配置开始训练:")
    print(json.dumps(config, indent=4, ensure_ascii=False))
    
    confirm = input("确认开始训练吗? (y/n): ").strip().lower()
    if confirm == 'y':
        try:
            command = ["python", "train.py", "--config_file", CONFIG_FILE]
            print(f"\n执行命令: {' '.join(command)}\n")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"训练过程中发生错误: {e}")
        except FileNotFoundError:
            print("错误: 'python' 命令未找到。请确保Python已安装并配置在系统路径中。")
    else:
        print("训练已取消。")


def main():
    """主菜单"""
    while True:
        print("\n请选择操作:")
        print("1. 交互式修改配置")
        print("2. 使用当前配置开始训练")
        print("3. 查看当前配置")
        print("4. 退出")
        
        choice = input("请输入选项 (1/2/3/4): ").strip()
        
        if choice == '1':
            interactive_configure()
        elif choice == '2':
            run_training()
        elif choice == '3':
            config = load_config()
            if config:
                print("\n当前配置:")
                print(json.dumps(config, indent=4, ensure_ascii=False))
        elif choice == '4':
            print("再见！")
            break
        else:
            print("无效的选项，请重新输入。")

if __name__ == "__main__":
    main()
