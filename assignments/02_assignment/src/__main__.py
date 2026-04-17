from task1 import main as task_1
from task2 import main as task_2
from task3 import main as task_3
from task3_benchmark import main as task_3_benchmark
from task4 import main as task_4
from task4_benchmark import main as task_4_benchmark

def main():
    tasks = {
        "Task 1": task_1,
        "Task 2": task_2,
        "Task 3": task_3,
        "Task 3 Benchmark": task_3_benchmark,
        "Task 4": task_4,
        "Task 4 Benchmark": task_4_benchmark
    }
    for name, task in tasks.items():
        print(f"Running {name}...")
        task()
        print(f"{name} completed!\n")

if __name__ == "__main__":
    main()
