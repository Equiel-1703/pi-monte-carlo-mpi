#include <mpi.h>

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINT_TAG 100

typedef struct
{
    int inside_samples_amount;
    int outside_samples_amount;
} Samples;

void initialize_message_buffer()
{
    int buff_size = MPI_BSEND_OVERHEAD + (10'000 * sizeof(char));
    void *buffer = malloc(buff_size);

    MPI_Buffer_attach(buffer, buff_size);
}

void delete_message_buffer()
{
    void *buffer;
    int buff_size;

    // Essa função é bloqueante. Ela só retorna quando todas as mensagens no buffer já foram enviadas.
    MPI_Buffer_detach(&buffer, &buff_size);

    free(buffer);
}

void ordered_print(char *str, int *counter)
{
    MPI_Bsend(str, strlen(str) + 1, MPI_CHAR, 0, PRINT_TAG, MPI_COMM_WORLD);
    (*counter) += 1;
}

// Gera um número aleatório entre 0 e 1 (inclusivo)
double randf()
{
    return (double)rand() / (double)RAND_MAX;
}

Samples calculate_samples(int samples_to_calculate)
{
    Samples s;
    s.inside_samples_amount = 0;
    s.outside_samples_amount = 0;

    for (int i = 0; i < samples_to_calculate; i++)
    {
        // Range of values are from -1 to +1
        double x = randf() * 2.0 - 1.0;
        double y = randf() * 2.0 - 1.0;

        double distance_from_origin = sqrt(x * x + y * y);

        if (distance_from_origin <= 1.0)
        {
            s.inside_samples_amount += 1;
        }

        // Um ponto sempre vai estar dentro do quadrado, independente de estar dentro do círculo ou não
        s.outside_samples_amount += 1;
    }

    return s;
}

int main(int argc, char const *argv[])
{
    MPI_Init(NULL, NULL);

    int rank, no_processes;
    int terminate = 0, samples_to_calculate = 0, samples_amount = 0;
    char send_buffer[101];
    int sent_message_count = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &no_processes);

    // Processo 0 vai fazer validações e configurar variáveis de execução
    if (rank == 0)
    {
        if (argc != 2)
        {
            printf("Por favor, informe o número de samples que você deseja usar para o cálculo:\n\n");
            printf("\tmpirun pi_monte_carlo.out <numero_de_samples>\n\n");

            terminate = 1; // Encerra execução
        }
        else
        {
            samples_amount = atoi(argv[1]);
            samples_to_calculate = samples_amount / no_processes;

            printf("- No. processos: %d\n", no_processes);
            printf("- Cada processo vai calcular %d samples de um total de %d.\n", samples_to_calculate, samples_amount);
        }
    }

    // Os processos verificam se tudo ocorreu bem
    MPI_Bcast(&terminate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (terminate)
    {
        MPI_Finalize();
        return 0;
    }

    // Inicializa buffer de mensagens de todos os processos para output ordenado no terminal
    initialize_message_buffer();

    // Solicita samples que este processo deve calcular
    MPI_Bcast(&samples_to_calculate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Processo 0 vai calcular o número de samples per process mais os restantes (caso a divisão não tenha sido exata)
    if (rank == 0)
    {
        samples_to_calculate += samples_amount - (samples_to_calculate * no_processes);
        sprintf(send_buffer, "- Processo 0 vai calcular %d samples\n", samples_to_calculate);
        ordered_print(send_buffer, &sent_message_count);
    }
    else
    {
        sprintf(send_buffer, "- Processo %d vai calcular %d samples\n", rank, samples_to_calculate);
        ordered_print(send_buffer, &sent_message_count);
    }

    // Multiplicando o rank por 4 e somando no time(0) para evitar que cada processo gere números aleatórios próximos/iguais
    srand(time(0) + (rank * 4));
    // Cada processo vai gerar e contar quantos pontos aleatórios ficaram dentro e fora do círculo unitário
    Samples s = calculate_samples(samples_to_calculate);
    int s_array[2] = {s.inside_samples_amount, s.outside_samples_amount};

    // Somar tudo no processo root
    int total_count[2];
    MPI_Reduce(s_array, total_count, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Processo root calcula o PI usando os samples obtidos
    double pi_aprox;
    if (rank == 0)
    {
        pi_aprox = 4.0 * (double)total_count[0] / (double)total_count[1];
    }

    sprintf(send_buffer, "Samples do rank %d: ins=%d out=%d\n", rank, s.inside_samples_amount, s.outside_samples_amount);
    ordered_print(send_buffer, &sent_message_count);

    // Verifica quantas mensagens o processo root (0) deve receber
    int total_messages;
    MPI_Reduce(&sent_message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int flag = 1;
        MPI_Status status;
        char receive_buffer[101];

        // Só encerra o loop quando não houver mais mensagens para processar
        while (total_messages > 0)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, PRINT_TAG, MPI_COMM_WORLD, &flag, &status);

            if (flag)
            {
                int msg_size;
                MPI_Get_count(&status, MPI_CHAR, &msg_size);
                MPI_Recv(receive_buffer, msg_size, MPI_CHAR, status.MPI_SOURCE, PRINT_TAG, MPI_COMM_WORLD, &status);

                printf("%s", receive_buffer);

                total_messages -= 1;
            }
        }

        printf("\nPI aproximado (%d samples) = %lf\n", samples_amount, pi_aprox);
        printf("Total de samples no círculo unitário: %d\n", total_count[0]);
        printf("Total de samples fora do círculo unitário: %d\n", total_count[1]);
    }

    delete_message_buffer();

    MPI_Finalize();

    return 0;
}
