/*
This code simulates a resonating n-dimensional space using a simplified
physical model.

The code iterates over an array representing an n-dimensional grid. At 
each step a new value is calculated for each point by considering two factors:

1) Each point continues moving in the same direction it was moving before
   (called "momentum"). An amount is added to the value of the point equal
   to the difference between the past two values of that point, times a 
   coefficient c_momentum (which may need to be < 1.0 to prevent divergence).

2) Each point is pulled on by its surrounding points (those whose coordinates
   are equal to this point plus or minus 1 in each of the n dimensions).
   The average of the difference between the surrounding points and the
   point being evaluated is multiplied by another coefficient, c_pull.

The results of each step are sampled at one or more coordinates in the grid
and these values are written to a file for later analysis to (hopefully)
turn into sound.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>


/*
We use a #define to specify what format of floating-point numbers we
will use. We have to store a value for each point on our grid for two
consecutive steps, so we will need 2 * (the product of the array sizes
in each dimension) * (the size of a floating point number) memory space.
Although double would be preferred for accurate calculations, a double
is 8 bytes while a float is 4 bytes. Choose wisely.
*/
#define MYFLT float

/*
When considering the surrounding points of a given point, the question
arises as to how to handle the edges. Three different possibilities,
called "end strategies", are implemented. These are

FIXED: The point just off the edge is always considered to have
       a fixed value of zero. This is like a string which is fixed
       at the edges.

WRAPPED: The point just off the edge is the value of the opposite
         edge, so the dimension is "wrapped around".

LOOSE: The point just off the edge is not used in the calculation.
       The average used to calculate the pull contains one less point.
*/
typedef enum {
  FIXED,
  WRAPPED,
  LOOSE
} end_strategy_t;


/*
The main structure defining the n-dimensional space.
*/
typedef struct {
  int dimcount;
  int *dimsize;
  MYFLT c_momentum, c_pull;
  end_strategy_t *left_strategy, *right_strategy;
  int bufsize;
  MYFLT *bufA, *bufB;
  int **adj_cache;
} physmod_t;

physmod_t *PHYSMOD;

void randomize_buffer(MYFLT *buf, int size, MYFLT lo, MYFLT hi);

/*
The number of dimensions is also in this #define, which is used for
the dimensions of some working arrays in calc_pull_part(). This will
have to be changed when the configuration data is read from a file,
but for now, make sure DIMCOUNT is the same as PHYSMOD->dimcount.
*/

#define DIMCOUNT (4)

/*
Initializes the PHYSMOD data structure which defines the space.
TODO: Replace this code with something that reads values from a file. 
*/
void
init(physmod_t *p)
{
  const int dimcount = DIMCOUNT;
  const int dimsize[] = { (5*6*7)/2, (4*6*7)/2, (4*5*7)/2, (5*6*7)/2 };
  const MYFLT c_momentum = 0.9;
  const MYFLT c_pull = 0.95;
  const end_strategy_t left_strategy[] = { WRAPPED, WRAPPED, WRAPPED, WRAPPED };
  const end_strategy_t right_strategy[] = { WRAPPED, WRAPPED, WRAPPED, WRAPPED };
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
  p->bufA = calloc(bufsize, sizeof(MYFLT));
  p->bufB = calloc(bufsize, sizeof(MYFLT));
  p->adj_cache = calloc(bufsize, sizeof(int *));
  randomize_buffer(p->bufA, bufsize, -1.0, 1.0);
}

MYFLT
random_between(MYFLT lo, MYFLT hi)
{
  MYFLT rand01;
  rand01 = (MYFLT)(random()) / (MYFLT)(RAND_MAX);
  return lo + (rand01 * (hi - lo));
}

void
randomize_buffer(MYFLT *buf, int size, MYFLT lo, MYFLT hi)
{
  int i;
  for (i=0 ; i<size ; i++) {
    buf[i] = random_between(lo, hi);
  }
}

/*
In order to accommodate an arbitrary number of dimensions, we
implement our own translation between a one-dimensional array
index and an n-dimensional set of coordinates. This translation
uses the length of each array dimension as defined in the 
PHYSMOD structure.

This function takes a pointer to the PHYSMOD struct, a 
one-dimensional index to the array, and an array to hold
corresponding n coordinates.
*/
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

/*
This function takes a pointer to the PHYSMOD struct and
an n-dimensional array of coordinates and returns the
corresponding one dimensional index.
*/
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


#define ADJ_IDX_FIXED (-1)
#define ADJ_IDX_IGNORE (-2)

/*
Calculate the buffer indices of all adjacent points to
a given buffer index c. The adjacent indices are placed in
in the passed-in adj_idx array, which is assumed to have
sufficent memory allocated to store 2*dimcount ints.

Negative numbers indicate special treatment to handle the
edges:

ADJ_IDX_FIXED (-1)
   Treat this point as always having the value zero
   (FIXED end stategy)

ADJ_IDX_IGNORE (-2)
   Omit this point when calculating the average pull
   (LOOSE end strategy)
*/
void
calc_adj(physmod_t *p, int c, int *adj_idx)
{
  int coords[DIMCOUNT], adj_coords[DIMCOUNT];
  int i;
  extract_coords(p, c, coords);
  /* points to the "left" in each dimension */
  for (i=0 ; i < p->dimcount ; i++) {
    adj_idx[i] = 0;
    /* start with a copy of this point's coordinates */
    memcpy(adj_coords, coords, p->dimcount * sizeof(int));
    /* move dimension i one step "to the left" */
    adj_coords[i] = coords[i]-1;
    /* handle cases where we went off the grid */
    if (adj_coords[i] < 0) {
      switch ((p->left_strategy)[i]) {
      case FIXED:
	adj_idx[i] = ADJ_IDX_FIXED;
	break;
      case WRAPPED:
	adj_coords[i] += (p->dimsize)[i];
	break;
      case LOOSE:
	adj_idx[i] = ADJ_IDX_IGNORE;
	break;
      }
    }
    if (adj_idx[i] == 0) {
      adj_idx[i] = combine_coords(p, adj_coords);
    }
  }
  /* points to the "right" in each dimension */
  for (i=0 ; i < p->dimcount ; i++) {
    adj_idx[p->dimcount + i] = 0;
    memcpy(adj_coords, coords, p->dimcount * sizeof(int));
    adj_coords[i] = coords[i]+1;
    if (adj_coords[i] >= (p->dimsize)[i]) {
      switch ((p->right_strategy)[i]) {
      case FIXED:
	adj_idx[p->dimcount + i] = ADJ_IDX_FIXED;
	break;
      case WRAPPED:
	adj_coords[i] -= (p->dimsize)[i];
	break;
      case LOOSE:
	adj_idx[p->dimcount + 1] = ADJ_IDX_IGNORE;
	break;
      }
    }
    if (adj_idx[p->dimcount + i] == 0) {
      adj_idx[p->dimcount + i] = combine_coords(p, adj_coords);
    }
  }
}


/*
Calculates the pull part of the simulation for one-dimensional
index c of buffer buf. 
*/
MYFLT
calc_pull_part(physmod_t *p, MYFLT *buf, int c)
{
  int *adj_idx;
  int i, adj_count;
  MYFLT pull_part, value, adj_value;
  /* if adj_cache is nonzero, we assume it's initialized and should be used */
  if (p->adj_cache) {
    /* if the adj_cache entry for this point is zero, we need to fill it */
    if (!p->adj_cache[c]) {
      p->adj_cache[c] = calloc(2 * p->dimcount, sizeof(int));
      calc_adj(p, c, p->adj_cache[c]);
    }
    /* use the cached values */
    adj_idx = p->adj_cache[c];
  } else {
    /* cache is not in use, calculate adj values (each time) */
    adj_idx = calloc(2 * p->dimcount, sizeof(int));
    calc_adj(p, c, adj_idx);
  }
  /* Now that we have all the adjacent points, calculate pull */
  adj_count = 0;
  pull_part = 0;
  for (i=0 ; i < 2*p->dimcount ; i++) {
    switch (adj_idx[i]) {
    case ADJ_IDX_FIXED:
      pull_part += (0 - buf[c]);
      adj_count++;
      break;
    case ADJ_IDX_IGNORE:
      /* do nothing */
      break;
    default:
      pull_part += (buf[adj_idx[i]] - buf[c]);
      adj_count++;
    }
  }
  pull_part /= adj_count;
  pull_part *= p->c_pull;
  return pull_part;
}

/*
Do one iteration of the resonance simulation.
Note: curr_buf will hold the current step in the simulation after
this step is complete (i.e. it is the buffer being written to).

On entry, curr_buf is expected to contain the data from two steps ago,
while prev_buf contains the data from one step ago. 
*/
void
do_step(physmod_t *p, MYFLT *prev_buf, MYFLT *curr_buf)
{
  MYFLT momentum_part, pull_part;
  int i;
  
#pragma omp parallel for private(i, momentum_part, pull_part) shared(p, prev_buf, curr_buf)
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

  for (i=0 ; i<100 ; i++) {
    do_step(PHYSMOD, PHYSMOD->bufA, PHYSMOD->bufB);
    do_step(PHYSMOD, PHYSMOD->bufB, PHYSMOD->bufA);
  }
}
