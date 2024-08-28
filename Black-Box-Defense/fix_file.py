def fix_file_path(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    fixed_lines = []
    for line in lines:
        fixed_line = line.replace('\\', '/')
        fixed_lines.append(fixed_line)
    
    with open(file_path, 'w') as file:
        file.writelines(fixed_lines)

# Sử dụng hàm với đường dẫn đến file của bạn
fix_file_path(r"D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\Dataset\SIPADMEK\process\train.txt")
fix_file_path(r"D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\Dataset\SIPADMEK\process\val.txt")
fix_file_path(r"D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\Dataset\SIPADMEK\process\test.txt")