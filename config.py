import json
import os
import subprocess
from copy import deepcopy

CONFIG_FILE = "config/config.json"  # 统一配置文件

SECTION_ORDER = ["trainConfig", "splitConfig", "preprocessConfig"]

LIST_FIELDS = {
    "preprocessConfig": ["convertFiles", "validLabels", "removeKeywords"],
}

def load_full_config():
    """加载完整配置 (包含 trainConfig / splitConfig / preprocessConfig)"""
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件 {CONFIG_FILE} 不存在。")
        return None
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_full_config(cfg):
    """保存完整配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {CONFIG_FILE}")

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

def edit_dict_section(name: str, section: dict):
    print(f"\n=== 编辑 {name} ===")
    modified = deepcopy(section)
    keys = list(modified.keys())
    if not keys:
        print("(该部分当前为空，直接输入新键创建。例如 key=value)")
    while True:
        print("\n当前键值:")
        for k in keys:
            v = modified[k]
            short = v if isinstance(v, (int,float,bool)) else ("<list>" if isinstance(v, list) else str(v)[:40])
            print(f"  - {k}: {short}")
        print("操作: [e 键]编辑  [a]新增键  [d 键]删除  [l 键]编辑列表  [q]保存退出  [x]放弃退出")
        op = input("选择操作: ").strip().lower()
        if op == 'q':
            return modified, True
        if op == 'x':
            return section, False
        if op.startswith('e '):
            k = op[2:].strip()
            if k in modified:
                modified[k] = get_input(f"修改 {k}", modified[k])
            else:
                print("键不存在")
        elif op == 'a':
            line = input("输入 新键=值: ").strip()
            if '=' in line:
                k,v = line.split('=',1)
                k=k.strip(); v=v.strip()
                if v.lower() in ['true','false']:
                    nv = v.lower()=='true'
                else:
                    try:
                        nv = int(v)
                    except ValueError:
                        try:
                            nv = float(v)
                        except ValueError:
                            nv = v
                modified[k]=nv
                if k not in keys: keys.append(k)
        elif op.startswith('d '):
            k = op[2:].strip()
            if k in modified:
                modified.pop(k)
                keys.remove(k)
        elif op.startswith('l '):
            k = op[2:].strip()
            if k not in modified:
                print("键不存在")
                continue
            if not isinstance(modified[k], list):
                print("该键不是列表")
                continue
            modified[k] = edit_list(k, modified[k])
        else:
            print("无效指令。示例: e learning_rate  或  l convertFiles")

def edit_list(name: str, current: list):
    print(f"\n--- 编辑列表 {name} (当前 {len(current)} 项) ---")
    while True:
        print(f"前若干项: {current[:5]}")
        print("操作: [r]替换  [a]追加  [d]按索引删除  [c]清空  [q]完成")
        op = input("选择: ").strip().lower()
        if op=='q':
            return current
        if op=='r':
            print("输入多行，新列表。空行结束。")
            lines=[]
            while True:
                line=input()
                if line.strip()=="":
                    break
                lines.append(line.strip())
            current = lines
        elif op=='a':
            items = input("输入要追加的项(逗号分隔): ").strip()
            if items:
                current.extend([x.strip() for x in items.split(',') if x.strip()])
        elif op=='d':
            idxs = input("输入要删除的索引(逗号分隔): ").strip()
            try:
                to_del = sorted({int(i) for i in idxs.split(',')}, reverse=True)
                for i in to_del:
                    if 0 <= i < len(current):
                        current.pop(i)
            except ValueError:
                print("索引格式错误")
        elif op=='c':
            confirm = input("确认清空? (y/n): ").strip().lower()
            if confirm=='y':
                current=[]
        else:
            print("未知操作")

def interactive_configure():
    """交互式配置统一文件中的各部分"""
    full = load_full_config()
    if full is None:
        return
    changed = False
    while True:
        print("\n=== 配置编辑器 ===")
        for i, sec in enumerate(SECTION_ORDER, 1):
            marker = "*" if sec in full else "!缺失"  # 标记
            print(f"{i}. {sec} {marker}")
        print(f"{len(SECTION_ORDER)+1}. 保存并退出")
        print(f"{len(SECTION_ORDER)+2}. 放弃退出")
        choice = input("选择要编辑的部分: ").strip()
        if choice == str(len(SECTION_ORDER)+1):
            if changed:
                save_full_config(full)
            print("退出。")
            return
        if choice == str(len(SECTION_ORDER)+2):
            print("放弃修改，退出。")
            return
        try:
            idx = int(choice)-1
        except ValueError:
            print("无效选项")
            continue
        if 0 <= idx < len(SECTION_ORDER):
            sec_name = SECTION_ORDER[idx]
            sec_dict = full.get(sec_name, {})
            new_sec, saved = edit_dict_section(sec_name, sec_dict)
            if saved and new_sec != sec_dict:
                full[sec_name] = new_sec
                changed = True
        else:
            print("无效索引")

def run_training():
    """使用当前统一配置启动训练 (仅展示 trainConfig)"""
    full = load_full_config()
    if full is None:
        return
    train_cfg = full.get('trainConfig') or {}
    print("\n即将使用 trainConfig: ")
    print(json.dumps(train_cfg, indent=2, ensure_ascii=False))
    confirm = input("确认开始训练? (y/n): ").strip().lower()
    if confirm!='y':
        print("取消。")
        return
    try:
        command = ["python", "train.py", "--config_file", CONFIG_FILE]
        print(f"\n执行命令: {' '.join(command)}\n")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练过程中发生错误: {e}")
    except FileNotFoundError:
        print("错误: 未找到 python 可执行文件。")


def main():
    while True:
        print("\n主菜单:")
        print("1. 编辑配置 (train/split/preprocess)")
        print("2. 查看完整配置")
        print("3. 运行训练 (使用 trainConfig)")
        print("4. 退出")
        choice = input("选择: ").strip()
        if choice=='1':
            interactive_configure()
        elif choice=='2':
            full = load_full_config()
            if full:
                print(json.dumps(full, indent=2, ensure_ascii=False))
        elif choice=='3':
            run_training()
        elif choice=='4':
            print("退出。")
            break
        else:
            print("无效选项。")

if __name__ == "__main__":
    main()
