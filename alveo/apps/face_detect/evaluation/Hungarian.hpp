/*
 *  C Implementation of Kuhn's Hungarian Method
 *  Copyright (C) 2003  Brian Gerkey <gerkey@robotics.usc.edu>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/*
 * A C implementation of the Hungarian method for solving Optimal Assignment
 * Problems.  Theoretically runs in O(mn^2) time for an m X n problem; this 
 * implementation is certainly not as fast as it could be.
 *
 * $Id: hungarian.h,v 1.8 2003/03/14 22:07:42 gerkey Exp $
 */

#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h> // for size_t
#include <string.h>

/* bzero is not always available (e.g., in Win32) */
#ifndef bzero
  #define bzero(b,len) (memset((b), '\0', (len)), (void) 0)
#endif

/* are we maximizing or minimizing? */
#define HUNGARIAN_MIN (0)
#define HUNGARIAN_MAX (1)
#define HUNGARIAN_EPS (1e-15)

/*
 * a simple linked list
 */
typedef struct
{
  unsigned int* i;
  unsigned int* j;
  unsigned int k;
} hungarian_sequence_t;

/*
 * we'll use objects of this type to keep track of the state of the problem
 * and its solution
 */
typedef struct
{
  size_t m,n;  // problem dimensions
  double** r;     // the rating (utility) matrix
  int** q;     // the Q matrix
  double* u;      // the U vector
  double* v;      // the V vector
  int* ess_rows; // list of essential rows
  int* ess_cols; // list of essential columns
  hungarian_sequence_t seq; // sequence of i's and j's
  double row_total, col_total; // row and column totals
  int* a;  // assignment vector
  double maxutil;  // maximum utility
  int mode; // are we maximizing or minimizing?
} hungarian_t;

/*
 * initialize the given object as an mXn problem.  allocates storage, which
 * should be freed with hungarian_fini().
 */
void hungarian_init(hungarian_t* prob, double** r, size_t m, size_t n, int mode);

/*
 * frees storage associated with the given problem object.  you must have
 * called hungarian_init() first.
 */
void hungarian_fini(hungarian_t* prob);

/*
 * solve the given problem.  runs the Hungarian Method on the rating matrix
 * to produce optimal assignment, which is stored in the vector prob->a.
 * you must have called hungarian_init() first.
 */
void hungarian_solve(hungarian_t* prob);

/*
 * prints out the resultant assignment in a 0-1 matrix form.  also computes
 * and prints out the benefit from the assignment.  you must have called
 * hungarian_solve() first.
 */
void hungarian_print_assignment(hungarian_t* prob);

/*
 * prints out the rating matrix for the given problem.  you must have called
 * hungarian_solve() first.
 */
void hungarian_print_rating(hungarian_t* prob);

/*
 * check whether an assigment is feasible.  returns 1 if the assigment is
 * feasible, 0 otherwise.  you must have called hungarian_solve() first.
 */
int hungarian_check_feasibility(hungarian_t* prob);

/*
 * computes and returns the benefit from the assignment.  you must have
 * called hungarian_solve() first.
 */
double hungarian_benefit(hungarian_t* prob);

#ifdef __cplusplus
}
#endif

#endif
