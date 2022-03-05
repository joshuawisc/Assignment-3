/**
 * Parallel VLSI Wire Routing via OpenMP
 * Joshua Mathews(jcmathew), Nolan Mass(nmass)
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <omp.h>

typedef struct { /* Define the data structure for wire here */
    int x[4];
    int y[4];
    int x1;
    int x2;
    int y1;
    int y2;
    int bend1x;
    int bend2x;
    int bend1y;
    int bend2y;
} wire_t;

typedef int cost_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif
