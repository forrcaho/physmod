/*
This code simulates a resonating n-dimensional space using a simplified
physical model.

The code iterates over two arrays representing the positions and velocities of
points moving in the (n+1)th dimension in an n-dimensional grid. At each step, new
values are calculated for each point by following these steps:

1) The position of each point is adjusted according to the velocity it has from
   the previous step. The velocity value times the coefficient coeff_momentum is
   added to the position of each point. 

2) The new velocity of each point is computed from the updated positions calculated
   in step (1). Each point is pulled on by its surrounding points (those whose 
   coordinates are equal to this point plus or minus 1 in each of the n dimensions).
   The sum of the difference between the surrounding points and the point being
   evaluated is multiplied by another coefficient, coeff_pull, to give the new velocity
   for that point.

The results of each step are sampled at one or more coordinates in the grid
and these values are written to a file for later analysis to (hopefully)
turn into sound.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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
       The sum used to calculate the pull contains one less point.
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
  MYFLT coeff_momentum, coeff_pull;
  end_strategy_t *left_strategy, *right_strategy;
  int bufsize;
  MYFLT *position, *velocity;
  int **adj_cache;
} physmod_t;

physmod_t *PHYSMOD;

void randomize_buffer(MYFLT *buf, int size, MYFLT lo, MYFLT hi);

/*
Initializes the PHYSMOD data structure which defines the space.
TODO: Replace this code with something that reads values from a file. 
*/
void
init(physmod_t *p)
{

  /*
  const int dimcount = 1;
  const int dimsize[] = { 64 };
  const MYFLT coeff_momentum = 1.0;
  const MYFLT coeff_pull = 1e-7;
  const end_strategy_t left_strategy[] = { FIXED };
  const end_strategy_t right_strategy[] = { FIXED };
  */

  const int dimcount = 4;
  const int dimsize[] = { (5*6*7)/2, (4*6*7)/2, (4*5*7)/2, (5*6*7)/2 };
  const MYFLT coeff_momentum = 1.0;
  const MYFLT coeff_pull = 1.0 + 1e-6;
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
  p->coeff_momentum = coeff_momentum;
  p->coeff_pull = coeff_pull; 
  p->bufsize = bufsize;
  p->position = calloc(bufsize, sizeof(MYFLT));
  p->velocity = calloc(bufsize, sizeof(MYFLT));
  p->adj_cache = calloc(bufsize, sizeof(int *));
  /* randomize_buffer(p->position, bufsize, -1.0, 1.0); */
  p->position[1] = 1; /* more like a pluck */
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
Calculate the one-dimensional buffer indices of all adjacent
points to a given one-dimensional buffer index c. The adjacent
indices are placed in in the passed-in adj_idx array, which is
assumed to have sufficent memory allocated to store 2*dimcount
ints.

A scratch array which can also hold 2*dimcount ints is also passed
in to this function. This is used in the conversion from a one-
diminsional to a multi-dimensional representation of coordinates.

By passing in a scratch array here we hope to avoid the deallocation
and reallocation of memory. We can't just declare this array locally
because its size is dependant on the number of dimensions we're
simulating.

Negative numbers indicate special treatment to handle the
edges:

ADJ_IDX_FIXED (-1)
   Treat this point as always having the value zero
   (FIXED end stategy)

ADJ_IDX_IGNORE (-2)
   Omit this point when calculating the total pull
   (LOOSE end strategy)
*/
void
calc_adj(physmod_t *p, int c, int *adj_idx, int *scratch)
{
  int *coords, *adj_coords;
  int i;

  /*
   The first dimcount ints of the scratch array hold the
   n-dimensional coordinates of the point we're evaluating
   whose one-dimensional index is c. This is assigned the array
   name "coords".

   The remaining dimcount ints of the scratch array hold the
   n-dimensional coordinates of an adjacent point in the array
   while we calculate its one-dimensional index. This is assigned
   the array name "adj_coords".
  */
  coords = scratch;
  adj_coords = scratch + p->dimcount;

  /* Put the n-dimensional coords for c in coords */
  extract_coords(p, c, coords);
  
  /* Calculate the adjacent points to the "left" in each dimension */
  for (i=0 ; i < p->dimcount ; i++) {

    /* 
    adj_idx[i] will hold the one-dimensional index for the point which
    is "to the left" in dimension i. We initially set this value to zero
    as a flag which let us distinguish "off the grid" cases.
    */
    adj_idx[i] = 0;

    /* 
    Now we put the n-dimensional coordinates for the adjacent point
    in adj_coords. We Start with a copy of this point's coordinates
    */
    memcpy(adj_coords, coords, p->dimcount * sizeof(int));
    /* 
    then we move one step "to the left" along dimension i. 
    */
    adj_coords[i] = coords[i]-1;
    
    /* 
    This is where we handle cases where we went "off the grid".
    FIXED and LOOSE end strategies require special handling when
    calculating the pull of adjacent points, so for these we use
    special negative one-dimensional index values that serve as 
    flags for that special handling. The WRAPPED end strategy requires 
    that we adjust the i-th dimension coordinate to wrap around, but
    then we will want to compute the one-dimensional index from the
    n-dimensional coordinates as usual.
    */
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
    /*
    If this n-dimensional point did not require a special end strategy 
    (it was not at an edge or the end strategy was WRAPPED) adj_idx[i]
    will still be 0, and now we need to translate from n-dimensional
    coordinates to a one-dimensional array index.
    */
    if (adj_idx[i] == 0) {
      adj_idx[i] = combine_coords(p, adj_coords);
    }
  }
  /* 
  Now, calculate points to the "right" in each dimension, following
  the same logic as we did for the "left" points above.
  */
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
Calculate the pull on one-dimensional index c of the position array.
Buffer scratch must have enough memory to store 2*p->dimcount ints. 
*/
MYFLT
calc_pull(physmod_t *p, MYFLT *position, int c, int *scratch)
{
  int *adj_idx;
  int i, adj_count;
  MYFLT pull, value, adj_value;

  /* 
  p->adj_cache is an optional cache of the one-dimensional indexes of the
  adjacent points to every point by its one-dimensional index. Special
  negative values are used for end strategies that require special handling.
  By caching this information we save a lot of computation for each iteration,
  but if memory is tight it doesn't have to be allocated.

  If the memory address p->adj_cache is nonzero, that means that memory has
  been allocated for it and we're using the cache. If the memory address
  p->adj_cache[c] for one-dimensional index c is zero, that means the cache
  for that index has not yet been populated so we allocate the memory and
  compute the adjacent indexes for that index.

  This means that, if the cache is in use, it should be fully populated in the
  first iteration of the simulation.
  */
  if (p->adj_cache) {
    if (!p->adj_cache[c]) {
      /* Populate the adjacent cache for this index, 
	 if it hasn't been done already */
      p->adj_cache[c] = calloc(2 * p->dimcount, sizeof(int));
      calc_adj(p, c, p->adj_cache[c], scratch);
    }
    /* Use the adjacent indexes from the cache. */
    adj_idx = p->adj_cache[c];
  } else {
    /* Cache is not in use, calculate adj values (each time) */
    adj_idx = calloc(2 * p->dimcount, sizeof(int));
    calc_adj(p, c, adj_idx, scratch);
  }
  /* 
  Now that we have all the adjacent points, we calculate the pull by
  summing the differences between the adjacent points and the point
  we're evaluating.
  */

  pull = 0;
  for (i=0 ; i < 2*p->dimcount ; i++) {
    switch (adj_idx[i]) {
    case ADJ_IDX_FIXED:
      /* A fixed point on the edge always stays at position zero. */
      pull += (0 - position[c]);
      break;
    case ADJ_IDX_IGNORE:
      /* An point off the edge with a LOOSE end strategy makes no
         contribution to the pull. There's nothing to do in this case. */
      break;
    default:
      pull += (position[adj_idx[i]] - position[c]);
    }
  }
  pull *= p->coeff_pull;
  return pull;
}

/*
Do one iteration of the resonance simulation.
Note: curr_buf will hold the current step in the simulation after
this step is complete (i.e. it is the buffer being written to).

On entry, curr_buf is expected to contain the data from two steps ago,
while prev_buf contains the data from one step ago. 

void
do_step(physmod_t *p, MYFLT *prev_buf, MYFLT *curr_buf, int *scratch)
{
  MYFLT momentum, pull;
  int i, thread_num;
  int *scratch_private;

#pragma omp parallel shared(p, prev_buf, curr_buf, scratch) \
  private(i, momentum, pull, thread_num, scratch_private)
  {
    thread_num = omp_get_thread_num();
    scratch_private = scratch + (2 * p->dimcount) * thread_num;
    #pragma omp for
    for (i=0 ; i < p->bufsize ; i++) {
      momentum = (prev_buf[i] - curr_buf[i]) * p->coeff_momentum;
      pull = calc_pull(p, prev_buf, i, scratch_private);
      curr_buf[i] = prev_buf[i] + momentum + pull;
    }
  }
}
*/

/*
Do one step of the resonance simulation.

First, the positions of each point are updated by their corresponding
velocities. 

Next, the velocities are adjusted by the pull calculated from
the differences in positions of each point from its adjacent points.

Note the OMP trickery here: to handle variable numbers of dimensions, we
dynamically allocate memory before we enter this parallel section, and
pass a unique buffer to each thread using its thread number. 
*/
void
do_step(physmod_t *p, int *scratch)
{
  MYFLT *position, *velocity;
  int i, thread_num;
  int *scratch_private;
  position = p->position;
  velocity = p->velocity;

#pragma omp parallel shared(position, velocity) \
 private(i, thread_num, scratch_private)
  {
    /* Assign each thred its own private scratch buffer */
    thread_num = omp_get_thread_num();
    scratch_private = scratch + (2 * p->dimcount) * thread_num;

    /* First, calculate the change in position of each point
       from the velocities */
#pragma omp for
    for (i=0 ; i < p->bufsize ; i++) {
      position[i] += velocity[i] * p->coeff_momentum;
    }

    /* Ensure all threads are done before the next step */
#pragma omp barrier

    /* Now, calculate the change in velocity of each point
       from the positions of adjacent points */
#pragma omp for
    for (i=0 ; i < p->bufsize ; i++) {
      velocity[i] += calc_pull(p, position, i, scratch_private);
    }
  }
}

int
main(int argc, char **argv)
{
  int c, i, j, tapA, tapB;
  int *scratch;
  int tapAcoords[1], tapBcoords[1];
  FILE *fpA, *fpB, *fpAt, *fpBt;
  srandom(time(NULL));
  PHYSMOD = calloc(1, sizeof(physmod_t));
  init(PHYSMOD);

  fpA = fopen("outA.dat", "wb");
  fpB = fopen("outB.dat", "wb");

  fpAt = fopen("outA.txt", "w");
  fpBt = fopen("outB.txt", "w");

  /* 
     The "taps" (locations where we record the output of our
     simulation) are the one place we still use a hardcoded
     number of dimensions. This must be fixed.
  */
  tapAcoords[0] = PHYSMOD->dimsize[0]/4;
  tapAcoords[1] = PHYSMOD->dimsize[1]/4;
  tapAcoords[2] = PHYSMOD->dimsize[2]/4;
  tapAcoords[3] = PHYSMOD->dimsize[3]/4;
  tapA = combine_coords(PHYSMOD, tapAcoords);
  
  tapBcoords[0] = 3*PHYSMOD->dimsize[0]/4;
  tapBcoords[1] = 3*PHYSMOD->dimsize[1]/4;
  tapBcoords[2] = 3*PHYSMOD->dimsize[2]/4;
  tapBcoords[3] = 3*PHYSMOD->dimsize[3]/4;
  tapB = combine_coords(PHYSMOD, tapBcoords);

  /* dynamically allocate scratch space for all threads */
  scratch = calloc(2 * PHYSMOD->dimcount * omp_get_max_threads(), sizeof(int));
  printf("max threads: %d\n", omp_get_max_threads());
  
  for (j=0 ; j<16; j++) {
    for (i=0 ; i<1024*64; i++) {
      do_step(PHYSMOD, scratch);
      fwrite( PHYSMOD->position + tapA, sizeof(MYFLT), 1, fpA);
      fprintf(fpAt, "%f\n", (PHYSMOD->position)[tapA]);
      fwrite( PHYSMOD->position + tapB, sizeof(MYFLT), 1, fpB);
      fprintf(fpBt, "%f\n", (PHYSMOD->position)[tapB]);
    }
    fflush(NULL);
    printf("%6d samples written\n", (j+1)*1024*64);
  }
  fclose(fpA);
  fclose(fpB);
}
