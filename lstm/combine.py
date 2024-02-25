# Updating the script to renumber the epochs in the second log file starting from 395

def renumber_epochs(log1_path, log2_path, start_epoch=395):
    # Step 1: Read the first log file and remove content from "epoch:395" onwards
    with open(log1_path, 'r') as file:
        log1_lines = file.readlines()
    # Find the line index for "epoch:395"
    epoch_start_line = next((i for i, line in enumerate(log1_lines) if line.strip().startswith(f'epoch:{start_epoch}')),
                            None)
    # Remove the lines from "epoch:395" onwards if found
    if epoch_start_line is not None:
        log1_lines = log1_lines[:epoch_start_line]

    # Step 2: Read the second log file and renumber epochs starting from 395
    with open(log2_path, 'r') as file:
        log2_lines = file.readlines()
    # Adjust the epoch number to start from 395
    for i in range(len(log2_lines)):
        if log2_lines[i].startswith('epoch:'):
            parts = log2_lines[i].split(',')
            # Extract the current epoch number and adjust it
            current_epoch = int(parts[0].split(':')[1])
            updated_epoch = current_epoch + start_epoch  # Add 395 to the current epoch number
            parts[0] = f'epoch:{updated_epoch}'
            log2_lines[i] = ','.join(parts)

    # Step 3: Combine both log contents
    combined_log_content = ''.join(log1_lines + log2_lines)

    # Step 4: Save the combined content to a new log file
    combined_log_path = 'renumbered_combined_log.log'
    with open(combined_log_path, 'w') as file:
        file.write(combined_log_content)

    return combined_log_path


# Call the function with the paths to the mock log files for demonstration
renumbered_combined_log_path = renumber_epochs('one_alpha6_2e2048.log', 'one_alpha6_2.1e2048.log', start_epoch=395)
print(renumbered_combined_log_path)
