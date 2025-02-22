import multiprocessing

def append_numbers(shared_list, lock):
    x = 1
    i = 0
    
    while True:
        with lock:  # Förhindra race conditions
            shared_list.append(x)
        
        i += 1
        if i % 10000 == 0:
            with lock:
                print("Listlängd:", len(shared_list))
            i = 0

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_list = manager.list()  # Delad lista mellan processer
    lock = manager.Lock()  # Lås för att förhindra race conditions

    processes = []
    num_processes = multiprocessing.cpu_count()

    for _ in range(num_processes):
        p = multiprocessing.Process(target=append_numbers, args=(shared_list, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
