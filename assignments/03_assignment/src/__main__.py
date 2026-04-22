from task1 import main as task_1
def main():
    tasks = {
        "Task 1": task_1,
    }
    for name, task in tasks.items():
        print(f"Running {name}...")
        task()
        print(f"{name} completed!\n")

if __name__ == "__main__":
    main()
