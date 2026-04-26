from task1 import main as task_1
from task2 import main as task_2
from task3 import task_3a
from task3 import task_3b

def main():
    tasks = {
        "Task 1": task_1,
        "Task 2": task_2,
        "Task 3a": task_3a,
        "Task 3b": task_3b,
    }
    for name, task in tasks.items():
        print(f"Running {name}...")
        task()
        print(f"{name} completed!\n")

if __name__ == "__main__":
    main()
