import os
from os.path import join, isdir, exists

#从lasot数据集中创建软链接到指定目录
def create_symlinks():
    # 数据集路径和软链接目录
    lasot_path = '/media/xyz/TY/DATA/LaSOTBenchmark'
    link_dir = os.path.expanduser('./data')  # 自动处理波浪号

    # 指定需要创建软链接的序列名称列表（按需修改）
    sequences = [
        "airplane-19",
        "basketball-17",
        "bear-18",
        "bicycle-13",
        "bird-10",
        "boat-16",
        "book-13",
        "bottle-10",
        "bus-9",
        "car-15",
        "cat-9",
        "cattle-5",
        "chameleon-1",
        "coin-2",
        "crab-4",
        "crocodile-20",
        "cup-19",
        "deer-13",
        "dog-5",
        "drone-12"
    ]


    # 检查数据集路径是否存在
    if not isdir(lasot_path):
        raise FileNotFoundError(f"数据集路径不存在: {lasot_path}")

    # 创建目标目录（如果不存在）
    os.makedirs(link_dir, exist_ok=True)

    found = set()
    # 遍历所有类别目录
    for class_name in os.listdir(lasot_path):
        class_path = join(lasot_path, class_name)
        if not isdir(class_path):
            continue

        # 遍历每个类别中的序列
        for seq_name in os.listdir(class_path):
            seq_path = join(class_path, seq_name)
            if isdir(seq_path) and seq_name in sequences:
                # 创建软链接路径
                link_path = join(link_dir, seq_name)

                # 避免重复创建
                if not exists(link_path):
                    os.symlink(seq_path, link_path)
                    print(f'创建成功: {link_path} -> {seq_path}')
                    found.add(seq_name)
                else:
                    print(f'已存在: {link_path}')

    # 检查未找到的序列
    not_found = set(sequences) - found
    if not_found:
        print("警告：以下序列未找到:", ", ".join(not_found))


if __name__ == '__main__':
    create_symlinks()