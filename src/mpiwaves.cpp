#include <mpi.h>
#include <omp.h>

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

/*********************************************************************************
Класс MPIGridFunc
этот класс отвечает за хранение сетки численного метода на узлах MPI

Содержимое:
    N[x,y,z] -- размер сетки численного метода

    D[x,y,z] -- размер куска сетки, который хранится на узле MPI
    O[x,y,z] -- смещение верхнего левого угла сетки в данном блоке

    B[x,y,z] -- координаты блока MPI
    G[x,y,z] -- размер MPI сетки

    data -- локальный кусок данных
    extern_data -- внешние данные, 6 граней:
        0 -- данные с блока слева   по OX (-1)
        1 -- данные с блока справа  по OX (+1)
        2 -- данные с блока сверху  по OY (-1)
        3 -- данные с блока снизу   по OY (+1)
        4 -- данные с блока спереди по OZ (-1)
        5 -- данные с блока сзади   по OZ (+1)

Методы:
    Init() -- задает параметры
    MPIGridFunc() -- конструкторы

    SyncMPI() -- синхронизирует extern_data

    Get[N,B,D,G,O]() -- возвращает параметры

    Get()           -- возвращает элемент сетки в глобальной индексации
                       или nan, если элемента нет ни в data, ни в extern_data
    GetLocalIndex() -- возвращает элемент сетки в локальной индексации

    Set()           -- устанавливает элемент сетки в глобальной индексации
    SetLocalIndex() -- устанавливает элемент сетки в локальной индексации
*********************************************************************************/ 
class MPIGridFunc{
    int Nx, Ny, Nz;
    int Dx, Dy, Dz, Ox, Oy, Oz;
    int Bx, By, Bz, Gx, Gy, Gz;
    
    std::vector<double> data;
    std::vector<double> extern_data[6];

public:
    void Init(int Nx, int Ny, int Nz, int Gx, int Gy, int Gz, int Bx, int By, int Bz){
        // задаем размер сетки численного метода
        this->Nx = Nx;
        this->Ny = Ny;
        this->Nz = Nz;

        // задаем размер сетки MPI
        this->Gx = Gx;
        this->Gy = Gy;
        this->Gz = Gz;

        // задаем номер блока
        this->Bx = Bx;
        this->By = By;
        this->Bz = Bz;

        // вычиляем размеры MPI блока 
        Dx = Nx / Gx;
        Dy = Ny / Gy;
        Dz = Nz / Gz;

        // вычисляем смещение
        Ox = Dx * Bx;
        Oy = Dy * By;
        Oz = Dz * Bz;

        // Выделяем память
        // под данные
        data.resize(Dx * Dy * Dz);

        // под грани по OX
        extern_data[0].resize(Dy*Dz);
        extern_data[1].resize(Dy*Dz);

        // под грани по OY
        extern_data[2].resize(Dx*Dz);
        extern_data[3].resize(Dx*Dz);
        
        // под грани по OZ
        extern_data[4].resize(Dx*Dy);
        extern_data[5].resize(Dx*Dy);
    }

    // несколько конструкторов для удобства
    MPIGridFunc() {}
    MPIGridFunc(int Nx, int Ny, int Nz, int Gx, int Gy, int Gz, int Bx, int By, int Bz){
        this->Init(Nx,Ny,Nz, Gx,Gy,Gz, Bx,By,Bz);
    }

    // синхронизация внешних данных
    void SyncMPI(MPI_Comm comm){
        // сначала копируем внутренние данные
        // они будут обменяны с соседями

        // сначала готовим данные к обмену

        // тут есть хитрость: когда у нас размер MPI сетки равен 1 по некоторой оси,
        // мы уже тут меняем данные местами, а ниже пересылки не произойдет

        // грани OX
        #pragma omp parallel for
        for(int j = 0; j < Dy; ++j){
            for(int k = 0; k < Dz; ++k){
                int i = 0, target = (Gx > 1? 0: 1);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];
                i = Dx-1; target = (Gx > 1? 1: 0);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];
            }
        }

        // грани OY
        #pragma omp parallel for
        for(int i = 0; i < Dx; ++i){
            for(int k = 0; k < Dz; ++k){
                int j = 0, target = (Gy > 1? 2: 3);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];
                j = Dy-1; target = (Gy > 1? 3: 2);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];
            }
        }

        // грани OZ
        #pragma omp parallel for
        for(int i = 0; i < Dx; ++i){
            for(int j = 0; j < Dy; ++j){
                int k = 0, target = (Gz > 1? 4: 5);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];
                k = Dz-1; target = (Gz > 1? 5: 4);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];
            }
        }

        // координаты нашего блока и цели
        int t_crd[3];
        int m_crd[3];
        
        // ранк и статус операции
        int my_rank;
        MPI_Status status;

        // получаем наш ранк и координаты
        MPI_Comm_rank(comm, &my_rank);
        MPI_Cart_coords(comm, my_rank, 3, m_crd);

        // if((m_crd[0] != Bx) || (m_crd[1] != By) || (m_crd[2] != Bz))
        //     std::cerr << "WAT?????" << std::endl;

        // целевые узлы MPI
        int target[6];

        // список смещений
        int delta[6][3] = {
            {-1,0,0},{1,0,0},
            {0,-1,0},{0,1,0},
            {0,0,-1},{0,0,1}
        };

        // для каждой грани вычисляем цели
        for(int i = 0; i < 6; i++){
            // вычисляем координаты
            t_crd[0] = m_crd[0] + delta[i][0];
            t_crd[1] = m_crd[1] + delta[i][1];
            t_crd[2] = m_crd[2] + delta[i][2];
            
            // получаем ранк цели
            MPI_Cart_rank(comm, t_crd, &target[i]);            
        }

        // отправка данных в три этапа
        // OX, OY, OZ

        // четные меняются сначала направо, затем налево
        // нечетные -- наоборот
        for(int axis = 0; axis < 3; axis++){
            // вычисляем четность
            int tp = (m_crd[axis]) % 2;
            // int tp = (axis == 0? Bx : (axis == 1? By : Bz)) % 2;

            for(int tmp = 0; tmp < 2; tmp++){
                tp = 1 - tp;

                // вычисляем номер цели, куда будем отправлять
                int target_idx = 2 * axis + (1 - tp);

                // вычисляем теги отправки и приема
                // в них зашиты номер ноды, ось, направление
                int send_tag = 100000 + my_rank * 100 + axis * 10 + tp;
                int recv_tag = 100000 + target[target_idx] * 100 + axis * 10 + (1-tp);
                
                // если отправка не на себя, то отправляем
                if(my_rank != target[target_idx]){                
                    MPI_Sendrecv_replace(&extern_data[target_idx][0],extern_data[target_idx].size(),
                        MPI_DOUBLE,target[target_idx],send_tag,target[target_idx],recv_tag,
                        comm,&status);
                }
            }
        }
    }

    // возвращает данные в локальной индексации
    double GetLocalIndex(int i, int j, int k){
        // грани OX и основное значение
        if((j >= 0)&&(j<Dy)&&(k>=0)&&(k<Dz)){
            if(i == -1)
                return extern_data[0][k + Dz*j];
            // вот тут возвращается основное значение
            if((i >= 0)&&(i < Dx))
                return data[k + Dz*(j + Dy*i)];
            if(i == Dx)
                return extern_data[1][k + Dz*j];
        }
        // грани OY
        if((i >= 0)&&(i<Dx)&&(k>=0)&&(k<Dz)){
            if(j == -1)
                return extern_data[2][k + Dz*i];
            if(j == Dy)
                return extern_data[3][k + Dz*i];
        }
        // грани OZ
        if((i >= 0)&&(i<Dx)&&(j>=0)&&(j<Dy)){
            if(k == -1)
                return extern_data[4][j + Dy*i];
            if(k == Dz)
                return extern_data[5][j + Dy*i];
        }
        // иначе nan
        return nan("");
    }

    // устанавливает значение в локальной индексации
    bool SetLocalIndex(int i, int j, int k, double v){
        // если мы вне нужной области, то выплевываем false
        if((i < 0)||(i >= Dx)||(j < 0)||(j >= Dy)||(k < 0)||(k >= Dz))
            return false;

        // иначе, устанавливаем значение и говорим, что всё прошло хорошо
        data[k + Dz*(j + Dy*i)] = v;
        return true;
    }

    // возвращает параметры
    int GetN(int i) {return (i == 0? Nx : (i == 1? Ny : Nz));}
    int GetB(int i) {return (i == 0? Bx : (i == 1? By : Bz));}
    int GetG(int i) {return (i == 0? Gx : (i == 1? Gy : Gz));}
    int GetD(int i) {return (i == 0? Dx : (i == 1? Dy : Dz));}
    int GetO(int i) {return (i == 0? Ox : (i == 1? Oy : Oz));}

    // установка и получение значения в глобальной индексации
    double Get(int i, int j, int k) {return GetLocalIndex(i - Ox, j - Oy, k - Oz);}
    bool Set(int i, int j, int k, double v) {return SetLocalIndex(i - Ox, j - Oy, k - Oz, v);}    
};

// отладочная функция для сетки печати на экран
// оставил, т.к. жалко удалять :)
void PrintMPIGridFunc(MPI_Comm comm, MPIGridFunc& u, int print_rank = -1, bool extended = false){
    // полная печать -- будут напечатаны все значения
    bool full_print = (print_rank == -1);
    
    // ранк узла, чьи данные будем печатать
    print_rank = (full_print? 0 : print_rank);

    // если extended, то расширяем глобальную сетку на один узел
    // чтобы проверить пересылки крайних слоев
    int ext = int(extended);

    // вычисляем наш ранк
    int rank;
    MPI_Comm_rank(comm, &rank);

    // трехмерный цикл для печати
    for(int i = 0 - ext; i < u.GetN(0) + ext; i++){
        if(print_rank == rank)
            std::cerr << "[" << std::endl;
        for(int j = 0 - ext; j < u.GetN(1) + ext; j++){
            if(print_rank == rank)
                std::cerr << "\t";
            for(int k = 0 - ext; k < u.GetN(2) + ext; k++){
                // получаем значение сетки и подменяем его на очень маленькое,
                // если оно равно nan
                double loc_value = u.Get(i,j,k);
                loc_value = (isnan(loc_value)? -1e300: loc_value);

                // значение для печати
                double glob_value = loc_value;
                if(full_print)
                    // если печать глобальная, то берем максимум с каждого узла
                    // очевидно, что останется необходимое значение
                    MPI_Reduce(&loc_value,&glob_value,1,MPI_DOUBLE,MPI_MAX,0,comm);

                // печатаем значение или вопросик, если его был nan
                if(print_rank == rank){
                    if(glob_value == -1e300)
                        std::cerr << "?" << "\t";
                    else
                        std::cerr << glob_value << "\t";
                }
            }
            if(print_rank == rank)
                std::cerr << std::endl;
        }
        if(print_rank == rank)
            std::cerr << "]" << std::endl;
    }
}


/*
В варианте 2:
OX -- граничное условие I-го рода
OY -- граничное условие I-го рода
OZ -- периодическое граничное условие

Если взять начальным условием функцию:
    phi(x,y,z) = sin(x) * sin(y) * cos(z),
то несложно вычислить аналитическое решение:
    u(x,y,z,t) = sin(x) * sin(y) * cos(z) * cos(sqrt(3) * t)
*/

// начальное условие
double Phi(double x, double y, double z){
    return cos(x) * cos(y) * cos(z);
}

// аналитическое решение
double UAnalytics(double x, double y, double z, double t){
    return Phi(x,y,z) * cos(sqrt(3) * t);
}


/*********************************************************************************
Основная программа

параметры:
    N -- размер глобальной сетки
    Gx, Gy, Gz -- размеры MPI сетки
    compute_metrics -- флаг вычисления метрики
*********************************************************************************/
int main(int argc, char* argv[]){
    // инициализация MPI
    MPI_Init(&argc, &argv);

    // отмечаем момент начала
    double time_start, time_stop;
    time_start = MPI_Wtime();

    // получаем временный ранг и число MPI узлов
    int rank, size;
    MPI_Comm comm;    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // проверяем количество аргументов
    if(argc <= 4){
        if(rank == 0)
            std::cout << "ruslixag's MPI waves" << std::endl 
                << "Usage: mpiwaves <N> <Gx> <Gy> <Gz>[ <compute_metrics>[ <frames>]]" << std::endl
                << "N -- computing grid size" << std::endl
                << "G[x,y,z] -- MPI grid size (Must be same size as number MPI nodes)" <<std::endl
                << "compute_metrics -- 1, if need to compute L_inf metrics. Optional." << std::endl
                << "frames -- frame number. Optional." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // готовимся к созданию нового мира
    // парсим размеры сетки из 2-4 аргументов командной строки
    int dim[3], period[3], reorder;
    int coord[3];
    dim[0]=atoi(argv[2]); dim[1]=atoi(argv[3]); dim[2]=atoi(argv[4]);
    period[0]=1; period[1]=1; period[2]=1;

    // проверяем, что размер сетки совпадает с выделенным количеством узлов MPI
    int req_size = dim[0] * dim[1] * dim[2];
    if(req_size != size){
        if(rank == 0)
            std::cout << "Please run with correct thread number (got " << size << " instead of " << req_size << ")" << std::endl;

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // создаем мир
    MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, 1, &comm);

    // получаем новый ранг и координаты узла в сетке
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coord);

    // парсим 5 аргумент командной строки, если он есть
    bool compute_metrics = false;
    if(argc > 5) compute_metrics = bool(atoi(argv[5]));

    // парсим размер сетки численного метода из 1го аргумента
    int N;

    N = atoi(argv[1]);

    if(rank == 0){
        std::cout << "N = " << N << std::endl;
        std::cout << "Gx = " << dim[0] << std::endl;
        std::cout << "Gy = " << dim[1] << std::endl;
        std::cout << "Gz = " << dim[2] << std::endl;

        // кидаем ошибку, если вычислительная сетка не делится на размеры MPI сетки
        if(N % dim[0]){
            std::cout << N << " %% " << dim[0] << " != 0" << std::endl;
            MPI_Abort(comm, 1);
        }
        if(N % dim[1]){
            std::cout << N << " %% " << dim[1] << " != 0" << std::endl;
            MPI_Abort(comm, 1);
        }
        if(N % dim[2]){
            std::cout << N << " %% " << dim[2] << " != 0" << std::endl;
            MPI_Abort(comm, 1);
        }
    }

    // размер области 
    double L = 4 * M_PI;

    // инициализируем параметры вычислительной сетки
    int Nx = N, Ny = N, Nz = N;
    double Lx = L, Ly = L, Lz = L;

    // граничные условия
    bool is_periodic_x = true, is_periodic_y = true, is_periodic_z = true;

    // вычисляем шаг сетки. Важно отметить, что мы "обрезаем" правые границы по каждой координате
    // так как они равны левому слою вне зависимости от типа граничных условий
    // поэтому мы можем их просто не хранить 
    double hx = Lx / Nx,
        hy = Ly / Ny,
        hz = Lz / Nz;

    // инициализируем дискретизацию по времени
    int frames = 20;
    if(argc > 6) frames = atoi(argv[6]);

    double ht = pow(fmin(hx,fmin(hy,hz)), 2) / 2;

    if(rank == 0)
        std::cout << "T = " << ht * frames << std::endl;

    // печать размера сетки
    // в Самарском сказано, что для устойчивости численного метода
    // нужно, чтобы tau < h
    if(rank == 0){
        std::cout << "h   = " << fmax(hx,fmax(hy,hz)) << std::endl
            << "tau = " << ht << std::endl;
        
        // если tau > h, то предупреждаем о последствиях
        if(ht > fmax(hx,fmax(hy,hz)))
            std::cout << "Warning! tau > h! This prog is gonna crash!!!" << std::endl;
    }

    // Создаем буфер для функции
    // т.к. нужно хранить всего три слоя (вычисляемый и два предыдущих)
    // то выделим памяти всего на три сеточных функции
    std::vector<MPIGridFunc> u;
    for(int i = 0; i <3; ++i)
        u.push_back(MPIGridFunc(Nx,Ny,Nz,dim[0],dim[1],dim[2],coord[0],coord[1],coord[2]));
    
    // начинаем вычисления
    if(rank == 0){
        std::cout << "Running" << std::endl;
    }
    for(int frame = 0; frame <= frames; frame++){
        // замеряем время
        double loc_t1, loc_t2;

        // печатаем в stderr дополнительную информацию о работе программы
        if(rank == 0)
            std::cerr << "Frame " << frame << std::endl;
        
        // вычисляем нужные индексы в массиве u
        // n -- новый слой (n+1)
        // n1 -- предыдущий слой (n)
        // n2 -- предпредыдущий слой (n-1)
        int n = frame % 3;
        int n1 = (frame-1) % 3;
        int n2 = (frame-2) % 3;

        // загружаем размеры блока
        int Dx = u[n].GetD(0);
        int Dy = u[n].GetD(1);
        int Dz = u[n].GetD(2);

        // загружаем смещение блока в сетке численного метода
        int Ox = u[n].GetO(0);
        int Oy = u[n].GetO(1);
        int Oz = u[n].GetO(2);

        // начинаем вычисление
        // распараллелим только внешний цикл for, тогда каждая нить будет получать
        // непрерывный кусочек данных
        loc_t1 = MPI_Wtime();

        double metrics = 0;
        #pragma omp parallel for
        for(int i = 0; i < Dx; ++i){
            for(int j = 0; j < Dy; ++j)
                for(int k = 0; k < Dz; ++k){
                    // готовим переменную для посчитанного значения
                    double value = 0;
                    double x = hx * (Ox + i), 
                           y = hy * (Oy + j),
                           z = hz * (Oz + k);

                    // если это нулевой кадр, то вычисляем значение
                    // из начального условия u[0](i,j,k) = phi(i,j,k)
                    if(frame == 0)
                        value = Phi(x,y,z);

                    // если это первый кадр, то вычисляем значение
                    // из начального условия du/dt[0](i,j,k) = 0
                    // отметим, что в сетке u лежат нужные нам значения
                    // функции phi, поэтому можно не вычислять их заново
                    if(frame == 1)
                        value = u[n1].GetLocalIndex(i,j,k) + pow(ht,2)/2 * (
                            (u[n1].GetLocalIndex(i+1,j,k) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i-1,j,k))/pow(hx,2)+
                            (u[n1].GetLocalIndex(i,j+1,k) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i,j-1,k))/pow(hy,2)+
                            (u[n1].GetLocalIndex(i,j,k+1) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i,j,k-1))/pow(hz,2)
                        );
                    
                    // если это не нулевой и не первый кадр, то вычисляем значение
                    // разностным методом
                    if(frame >= 2)
                        value = 2*u[n1].GetLocalIndex(i,j,k) - u[n2].GetLocalIndex(i,j,k) + pow(ht,2) * (
                            (u[n1].GetLocalIndex(i+1,j,k) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i-1,j,k))/pow(hx,2)+
                            (u[n1].GetLocalIndex(i,j+1,k) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i,j-1,k))/pow(hy,2)+
                            (u[n1].GetLocalIndex(i,j,k+1) - 2*u[n1].GetLocalIndex(i,j,k) + u[n1].GetLocalIndex(i,j,k-1))/pow(hz,2)
                        );

                    // проверяем граничные условия
                    // для всех условий первого рода явно задаем значение равное нулю
                    // напомним, что в своей сетке мы отрезали правые границы
                    if(!is_periodic_x && (x == 0))
                        value = 0;
                    if(!is_periodic_y && (y == 0))
                        value = 0;
                    if(!is_periodic_z && (z == 0))
                        value = 0;

                    // сохраняем значение в сетку
                    bool flag = u[n].SetLocalIndex(i,j,k,value);

                    if(compute_metrics){
                        double loc_metrics = fabs(value - UAnalytics(x,y,z,frame*ht));
                        // если значение метрики меньше, чем разность в текущей точке
                        // то меняем метрику
                        // добавляем разницу -- это косыль, чтобы использовать допустимую операцию для atomic
                        double metrics_delta = loc_metrics - metrics;
                        if(metrics_delta > 0){
                            #pragma omp atomic
                            metrics += metrics_delta;
                        }
                    }
                }
        }

        // выводим время вычислений в stderr
        loc_t2 = MPI_Wtime();
        if(rank == 0)
            std::cerr << "\tComputing: " << loc_t2 - loc_t1 << std::endl;

        // выполняем синхронизацию, пересылая 6 граней между блоками
        loc_t1 = MPI_Wtime();

        u[n].SyncMPI(comm);

        loc_t2 = MPI_Wtime();
        if(rank == 0)
            std::cerr << "\tSync: " << loc_t2 - loc_t1 << std::endl;

        // теперь вычисляем глобальную метрику с помощью Reduce

        double result = 0;
        MPI_Reduce(&metrics,&result,1,MPI_DOUBLE,MPI_MAX,0,comm);            
        loc_t2 = MPI_Wtime();
        if(rank == 0)
            std::cerr << "\tComputing metrics: " << loc_t2 - loc_t1 << std::endl;

        // печатаем метрику в stdout на первом узле
        if(rank == 0)
            std::cout << "Frame " << frame << ": metrics = " << result << std::endl;

    }

    // печатаем информацию о работе программы
    time_stop = MPI_Wtime();
    if(rank == 0)
        std::cout << "Finished!" << std::endl
            << "Total frames: " << frames << std::endl
            << "Elapsed time: " << (time_stop - time_start) << std::endl
            << "Avg time per frame: " << (time_stop - time_start) / double(frames) << std::endl;

    // завершаем работу программы
    MPI_Finalize();

    // возвращаяем нуль
    return 0;
}