import sys, os
def print_progress_bar(label, current, total, bar_length=40):
    # Calculate progress as a float
    progress = current / total
    # Create filled part of the bar
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    # Print the bar with progress percentage
    sys.stdout.write(f"\r|{bar}| {progress:.1%} ({current}/{total}) - {label}")
    sys.stdout.flush()

def printFlush(text):
    sys.stdout.write(f"\r{text}")
    sys.stdout.flush()

def cleanUp(file_paths):
    # List of file paths to check and delete
    pctr       = 0
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted.")
            
        else:
            print(f"{file_path} does not exist.")

        pctr+=1

    print_progress_bar("Cleaning logs & files.", pctr, len(file_paths))

def ask_yes_or_no(question):
    while True:
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer in ['yes', 'y']:
            return True
        elif answer in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")