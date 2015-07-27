#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef enum {
  FIXED,
  WRAPPED,
  LOOSE
} end_strategy_t;

typedef struct {
  int dimcount;
  int *dimsize;
  double c_momentum, c_pull;
  end_strategy_t *left_strategy, *right_strategy;
  int bufsize;
  double *bufA, *bufB;
} physmod_t;

physmod_t *PHYSMOD;

void randomize_buffer(double *buf, int size, double lo, double hi);

#define DIMCOUNT (4)

/* TODO: Replace this code with something that reads values from a file. */
void
init(physmod_t *p)
{
  const int dimcount = DIMCOUNT;
  const int dimsize[] = { 5*6*7, 4*6*7, 4*5*7, 4*5*6 };
  const double c_momentum = 0.9;
  const double c_pull = 0.95;
  const end_strategy_t left_strategy[] = { WRAPPED, WRAPPED, WRAPPED, WRAPPED, WRAPPED };
  const end_strategy_t right_strategy[] = { WRAPPED, WRAPPED, WRAPPED, WRAPPED, WRAPPED };
  int i, bufsize;

  p->dimcount = dimcount;
  p->dimsize = calloc(dimcount, sizeof(int));
  p->left_strategy = calloc(dimcount, sizeof(end_strategy_t));
  p->right_strategy = calloc(dimcount, sizeof(end_strategy_t));
  bufsize = 1;
  for (i = 0 ; i < dimcount ; i++) {
    (p->dimsize)[i] = dimsize[i];
    (p->left_strategy)[i] = left_strategy[i];
    (p->right_strategy)[i] = right_strategy[i];
    bufsize *= dimsize[i];
  }
  p->c_momentum = c_momentum;
  p->c_pull = c_pull;
  p->bufsize = bufsize;
  p->bufA = calloc(bufsize, sizeof(double));
  p->bufB = calloc(bufsize, sizeof(double));
  randomize_buffer(p->bufA, bufsize, -1.0, 1.0);
}

double
random_between(double lo, double hi)
{
  double rand01;
  rand01 = (double)(random()) / (double)(RAND_MAX);
  return lo + (rand01 * (hi - lo));
}

void
randomize_buffer(double *buf, int size, double lo, double hi)
{
  int i;
  for (i=0 ; i<size ; i++) {
    buf[i] = random_between(lo, hi);
  }
}

void
extract_coords(physmod_t *p, int c, int *coords)
{
  int i;
  for (i = p->dimcount - 1 ; i > 0 ; i-- ) {
    coords[i] = c % (p->dimsize)[i];
    c = (c-coords[i]) / (p->dimsize)[i];
  }
  coords[0] = c;
}

int
combine_coords(physmod_t *p, int *coords)
{
  int dimsize, i, c;
  c = coords[0];
  for (i=1 ; i < p->dimcount ; i++) {
    c *= (p->dimsize)[i];
    c += coords[i];
  }
  return c;
}

double
calc_pull_part(physmod_t *p, double *buf, int c)
{
  int i, j, adj_c, adj_value_fixed, adj_value_ignored, adj_count;
  int coords[DIMCOUNT], adj_coords[DIMCOUNT];
  double adj_values[2*DIMCOUNT];
  double value, pull_part;
  value = buf[c];
  extract_coords(p, c, coords);

  adj_count = 0;
  /* points to the "left" in each dimension */
  for (i=0 ; i < p->dimcount ; i++) {
    adj_value_fixed = 0;
    adj_value_ignored = 0;
    for (j=0 ; j < p->dimcount ; j++) {
      if (i == j) {
	adj_coords[j] = coords[j]-1;
	if (adj_coords[j] < 0) {
	  switch ((p->left_strategy)[j]) {
	  case FIXED:
	    adj_value_fixed = 1;
	    break;
	  case WRAPPED:
	    adj_coords[j] += (p->dimsize)[j];
	    break;
	  case LOOSE:
	    adj_value_ignored = 1;
	    break;
	  }
	}
      } else {
	adj_coords[j] = coords[j];
      }
    }
    if (!adj_value_ignored) {
      if (adj_value_fixed) {
	adj_values[adj_count] = 0;
      } else {
	adj_c = combine_coords(p, adj_coords);
	adj_values[adj_count] = buf[adj_c];
      }
      adj_count++;
    }
  }
  /* points to the "right" in each dimension */
  for (i=0 ; i < p->dimcount ; i++) {
    adj_value_fixed = 0;
    adj_value_ignored = 0;
    for (j=0 ; j < p->dimcount ; j++) {
      if (i == j) {
	adj_coords[j] = coords[j]+1;
	if (adj_coords[j] >= (p->dimsize)[j]) {
	  switch ((p->left_strategy)[j]) {
	  case FIXED:
	    adj_value_fixed = 1;
	    break;
	  case WRAPPED:
	    adj_coords[j] -= (p->dimsize)[j];
	    break;
	  case LOOSE:
	    adj_value_ignored = 1;
	    break;
	  }
	}
      } else {
	adj_coords[j] = coords[j];
      }
    }
    if (!adj_value_ignored) {
      if (adj_value_fixed) {
	adj_values[adj_count] = 0;
      } else {
	adj_c = combine_coords(p, adj_coords);
	adj_values[adj_count] = buf[adj_c];
      }
      adj_count++;
    }
  }
  /* Now that we have all the adjacent points, calculate pull */
  pull_part = 0;
  for (i=0 ; i<adj_count ; i++) {
    pull_part += (adj_values[i] - value); 
  }
  pull_part /= adj_count;
  pull_part *= p->c_pull;
  return pull_part;
}

void
do_step(physmod_t *p, double *prev_buf, double *curr_buf)
{
  double momentum_part, pull_part;
  int i;
  for (i=0 ; i < p->bufsize ; i++) {
    momentum_part = (prev_buf[i] - curr_buf[i]) * p->c_momentum;
    pull_part = calc_pull_part(p, prev_buf, i);
    curr_buf[i] = prev_buf[i] + momentum_part + pull_part;
  }
}

int
main(int argc, char **argv)
{
  int c, i;
  int coords[5];
  srandom(time(NULL));
  PHYSMOD = calloc(1, sizeof(physmod_t));
  init(PHYSMOD);

  do_step(PHYSMOD, PHYSMOD->bufA, PHYSMOD->bufB);
  do_step(PHYSMOD, PHYSMOD->bufB, PHYSMOD->bufA);
}
