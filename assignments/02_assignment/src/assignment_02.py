from task1 import main as task_1
from task2 import main as task_2
from task3 import main as task_3
from task4 import main as task_4

def main():
    tasks = [task_1, task_2, task_3, task_4]
    for i, task in enumerate(tasks, 1):
        print(f"Running Task {i}...")
        task()
        print(f"Task {i} completed!\n")

if __name__ == "__main__":
    main()
