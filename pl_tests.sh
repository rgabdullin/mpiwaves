#!/bin/bash
set -e

# пути для логов
LOG_PATH="./logs"

# убиваем все имеющиеся задачи у пользователя
# bkill -u $(whoami) 0 || echo "No jobs"

# очищаем папку с логами
# echo "cleaning"
# rm -rf $LOG_PATH/*

# список параметров: размеры mpi-сеток и максимальное время работы
grid_array=("1 1 1" "1 1 2" "1 2 2" "2 2 2")

echo "computing"

# считаем для перечисленных N
# for N in 128 256 512
for N in 128 256
do
    # и для всех параметров
    for i in 0 1 2 3
    do
        # размеры mpi-сетки
        grid=${grid_array[i]}

        # путь к файлам stdout и stderr 
        log_filename="${N}_$(echo ${grid} | tr " " "x")"

        # число mpi нод
        num_nodes=$(( $(echo ${grid} | tr " " "*") ))

        # печатаем информацию о задаче
        echo "================================================"
        echo "nodes=$num_nodes"
        echo "grid=$grid"
        echo "log=$log_filename"

        # сабмитим задачу в очередь
        # bsub -R "affinity[socket(1)]" -n $num_nodes -o "$LOG_PATH/${log_filename}.out" -e "$LOG_PATH/${log_filename}.err" ./wrapper.sh $N $grid
        bsub -q normal -R "span[hosts=1] affinity[core(1,same=socket)]" -n $num_nodes -o "$LOG_PATH/${log_filename}.out" -e "$LOG_PATH/${log_filename}.err" ./wrapper.sh $N $grid
    done
    # спим 10 секунд
    sleep 3
done
