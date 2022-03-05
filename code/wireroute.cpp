/**
 * Parallel VLSI Wire Routing via OpenMP
 * Joshua Mathews(andrew_id 1), Nolan Mass(nmass)
 */

#include "wireroute.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <string>

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}

void serial(wire_t *wires, cost_t *costs, int num_of_wires, int dim_x, int dim_y) {
    // Iterate over wires
    int min_max_cost;
    int min_sum_cost;
    int bendx, bendy;
    
    for (int i = 0; i < num_of_wires; i++) {
        // Iterate horizontally, then vertically
        int x_start = std::min(wires[i].x[0], wires[i].x[3]);
        int x_end = std::max(wires[i].x[0], wires[i].x[3]);
        for (int x = x_start; x < x_end; x++) {
            std::pair<int, int> result = checkcost(wires[i], x, wires[i].y[0]);
            if (min_max_cost > result.first
                    || (min_sum_cost > result.second && min_max_cost == result.first)) {
                
                min_max_cost = result.first;
                min_sum_cost = result.second;
                bendx = x;
                bendy = wires[i].y[0];
            }
        }

        // Iterate vertically
        int y_start = std::min(wires[i].y[0], wires[i].y[3]);
        int y_end = std::max(wires[i].y[0], wires[i].y[3]);
        for (int y = y_start; y < y_end; y++) {
            std::pair<int, int> result = checkcost(wires[i], wires[i].x[0], y);
            if (min_max_cost > result.first
                    || (min_sum_cost > result.second && min_max_cost == result.first)) {
                
                min_max_cost = result.first;
                min_sum_cost = result.second;
                bendx = wires[i].x[0];
                bendy = y;
            }
        }

        wires[i].x[1] = bendx;
        wires[i].y[1] = bendy;
        if (bendx = wires[i].x[0]) {
            wires[i].x[2] = wires[i].x[3];
            wires[i].y[2] = bendy;
        } else {
            wires[i].x[2] = bendx;
            wires[i].y[2] = wires[i].y[3];
        }
    }
    
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    double SA_prob = get_option_float("-p", 0.1f);
    int SA_iters = get_option_int("-i", 5);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Probability parameter for simulated annealing: %lf.\n", SA_prob);
    printf("Number of simulated annealing iterations: %d\n", SA_iters);
    printf("Input file: %s\n", input_filename);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int dim_x, dim_y;
    int num_of_wires;

    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    fscanf(input, "%d\n", &num_of_wires);
    printf("%dx%d, %d wires\n", dim_x, dim_y, num_of_wires);
    wire_t *wires = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
    /* Read the grid dimension and wire information from file */
    for (int i = 0 ; i < num_of_wires ; i++) {
        fscanf(input, "%d %d %d %d\n", &(wires[i].x[0]), &(wires[i].y[0]), &(wires[i].x[3]), &(wires[i].y[3]));
        // printf("Wire %d: %d %d %d %d\n", i, (wires[i].x1), (wires[i].x2), (wires[i].y1), (wires[i].y2));
    }

    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    /* Initialize cost matrix */


    /* Initailize additional data structures needed in the algorithm */

    /* Conduct initial wire placement */
    for (int i = 0 ; i < num_of_wires ; i++) {
        // bend 1 is (x1, y2)
        wires[i].x[1] = wires[i].x[0];
        wires[i].y[1] = wires[i].y[3];

        // bend 2 is endpoint
        wires[i].x[2] = wires[i].x[3];
        wires[i].y[2] = wires[i].y[3];
    }
    
    /*Set wire costs*/
    for (int i = 0 ; i < num_of_wires ; i++) {
        // p is the start point, p+1 is the end point
        for (int p = 0; p < 3; p++) {
            int x1 = wires[i].x[p];
            int y1 = wires[i].y[p];
            int x2 = wires[i].x[p+1];
            int y2 = wires[i].y[p+1];
            if (x1 == x2) {
                int start = std::min(y1, y2);
                int end = std::max(y1, y2);
                int x = x1;
                for (int y = start; y <= end; y++) {
                    costs[y*dim_x + x]++;
                }
            }
            else { // y1 == y2
                int start = std::min(x1, x2);
                int end = std::max(x1, x2);
                int y = y1;
                for (int x = start; x <= end; x++ ) {
                    costs[y*dim_x + x]++;
                }
            }
        }
        costs[wires[i].y[1]*dim_x + wires[i].x[1]]--;
        costs[wires[i].y[2]*dim_x + wires[i].x[2]]--;
    }

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;

    /**
     * Implement the wire routing algorithm here
     * Feel free to structure the algorithm into different functions
     * Don't use global variables.
     * Use OpenMP to parallelize the algorithm.
     */
    // Function to run the algorithm
    serial(wires, costs, num_of_wires, dim_x, dim_y);

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    /* Write wires and costs to files */
    std::string input_filename_stripped = std::string(input_filename).substr(17, std::string(input_filename).length() - 17 - 4);
    std::string costs_filename = "costs_" + input_filename_stripped + "_" + std::to_string(num_of_threads) + ".txt";
    std::string wires_filename = "output_" + input_filename_stripped + "_" + std::to_string(num_of_threads) + ".txt";
    FILE *fpcosts = fopen(costs_filename.c_str(), "w+");
    FILE *fpwires = fopen(wires_filename.c_str(), "w+");

    // write costs
    fprintf(fpcosts, "%d %d\n", dim_x, dim_y);
    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {
            fprintf(fpcosts, "%d ", costs[y*dim_x + x]);
        }
        fprintf(fpcosts, "\n");
    }

    // write wires
    fprintf(fpwires, "%d %d\n%d\n", dim_x, dim_y, num_of_wires);
    for (int i = 0; i < num_of_wires ; i++) {
        for (int p = 0; p < 4; p++) {
            fprintf(fpwires, "%d %d ", wires[i].x[p], wires[i].y[p]);
        }
        fprintf(fpwires, "\n");
    }

    // fclose(fpcosts);
    // fclose(fpwires);
    return 0;
}
