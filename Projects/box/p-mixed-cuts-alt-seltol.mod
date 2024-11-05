/*********************************************
 * OPL 22.1.1.0 Model
 * Author: songkomkrit
 * Creation Date: Nov 4, 2024 at 1:15:57 AM
 *********************************************/

/*********************************************
 * DATA INFORMATION (INPUTS)
 *********************************************/
int dimold = ...;	// old dimension
int dimcontold = ...;	// old continuous dimension
int dim = ...;	// new dimension
int dimcont = ...;	// new continuous dimension
//int dimcat = ...;	// categorical dimension
int N = ...;	// number of instances
int n = ...;	// number of classes

/*********************************************
 * FEATURE SELECTION (INPUTS)
 *********************************************/
int seltol = ...;	// given number of total selected cont/cat dimensions (at most)
int selcont = ...;	// UB on number of selected continuous dimensions
int exccont = ...;	// computed LB on number of excluded continuous dimensions

/*********************************************
 * INDEX RANGES 1
 *********************************************/
range DS = 1..dim;		// for dimensions
range DSCONTOLD = 1..dimcontold;	// for old continuous dimensions
range DSCONT = 1..dimcont;	// for new continuous dimensions
range DSCAT = dimcont+1..dim;	// for shifted categorical dimensions
range IS = 1..N;		// for instances
range KS = 0..n;		// for classes

/*********************************************
 * INITIAL PARAMETERS (INPUTS)
 *********************************************/
float M[DS] = ...;		// big-M for all new/shifted dimensions (continuous and categorical)
float m[DSCONT] = ...;	// small-m for new continuous dimensions

/*********************************************
 * DATA EXTRACTION (INPUTS)
 *********************************************/
float xcontold[IS][DSCONTOLD] = ...;	// instances along old continuous dimensions
int xcat[IS][DSCAT] = ...;	// instances along shifted categorical dimensions
int y[IS] = ...;		// targets
int maxlab[DSCAT] = ...;		// maximum labels for new categorical dimensions
int p[DS] = ...;		// number of cuts along axes
int coef[DS] = ...;		// product coefficients

/*********************************************
 * NUMBER OF BOXES
 *********************************************/
int B = 1;				// initialize the number of boxes
execute {
	for (var j in DS)
		B = B*(p[j]+1);	// compute the number of boxes
}

/*********************************************
 * INDEX RANGES 2
 *********************************************/
range BS = 0..B-1;		// for regions

/*********************************************
 * TUPLES
 *********************************************/
tuple ContPairType {	// index for continuous cut
	int j;
	int q;
};

{ContPairType} ContPairs = {<j, q> | j in DSCONT, q in 0..p[j]+1};

tuple ContTripleType {	// index for continuous cut of each individual instance
	int i;
	int j;
	int q;
};

{ContTripleType} ContTriples = {<i, j, q> | i in IS, j in DSCONT, q in 0..p[j]};

tuple CatPairType {	// index for categorical group
	int j;
	int l;
};

{CatPairType} CatPairs = {<j, l> | j in DSCAT, l in 0..maxlab[j]};

/*********************************************
 * DECISION VARIABLES
 *********************************************/
dvar float l[ContTriples];
dvar float r[ContTriples];
dvar float bc[ContPairs];		// bc is in R (c = cut)
// Note that b is used for beta indexing
dvar float h[BS];			// h
dvar boolean a[ContTriples];	// alpha
dvar int+ v[CatPairs];	// v (categorical features)
dvar boolean g[IS][BS];		// gamma
dvar boolean z[BS][KS];		// 
// Feature selection
dvar boolean ccont[DSCONT][DSCONTOLD];	// select continuous dimensions
dvar boolean f[DSCAT];				// select categorical dimensions

/*********************************************
 * OBJECTIVE FUNCTION
 *********************************************/
minimize sum(b in BS) h[b];			// min total number of misclassifed instances

/*********************************************
 * CONSTRAINTS
 *********************************************/
subject to {

	forall(j in DSCONT)
		getnewcont:
			sum(jold in DSCONTOLD) ccont[j][jold] <= 1;

	forall(jold in DSCONTOLD)
		seloldcont:
			sum(j in DSCONT) ccont[j][jold] <= 1;

	forall(j in DSCONT, q in 0..p[j])
		bc[<j,q+1>] - bc[<j,q>] >= 0;

	forall(i in IS, j in DSCONT) {
		lbound:
	  		(sum(jold in DSCONTOLD) xcontold[i][jold]*ccont[j][jold]) - (sum(q in 0..p[j]) l[<i,j,q>]) >= 0;	
	  	rbound:
	  		(sum(jold in DSCONTOLD) xcontold[i][jold]*ccont[j][jold]) - (sum(q in 0..p[j]) r[<i,j,q>]) <= 0;
	}
	
	forall(i in IS, j in DSCONT, q in 0..p[j]) {
		l[<i,j,q>] + M[j]*a[<i,j,q>] >= 0;	
		l[<i,j,q>] - M[j]*a[<i,j,q>] <= 0;	
		l[<i,j,q>] - bc[<j,q>] + M[j]*a[<i,j,q>] <= M[j] + m[j];
		l[<i,j,q>] - bc[<j,q>] - M[j]*a[<i,j,q>] >= -M[j] + m[j];
		r[<i,j,q>] + M[j]*a[<i,j,q>] >= 0;	
		r[<i,j,q>] - M[j]*a[<i,j,q>] <= 0;	
		r[<i,j,q>] - bc[<j,q+1>] + M[j]*a[<i,j,q>] <= M[j] - m[j];
		r[<i,j,q>] - bc[<j,q+1>] - M[j]*a[<i,j,q>] >= -M[j] - m[j];				
	}
	
	forall(i in IS)
	  	(sum(j in DSCONT) coef[j]*(sum(q in 0..p[j]) q*a[<i,j,q>])) + (sum(j in DSCAT) coef[j]*v[<j,xcat[i][j]>]) - (sum(b in BS) b*g[i][b]) == 0;
	
	forall(i in IS, j in DSCONT)
		pregion:
			sum(q in 0..p[j]) a[<i,j,q>] == 1;
	
	forall(i in IS) {
		bregion:
			sum(b in BS) g[i][b] == 1;
	}
	
	forall(b in BS, k in KS)
		error1:
			h[b] + (sum(i in IS) (y[i] == k)*g[i][b]) + N*z[b][k] >= 0;
	
	forall(b in BS)
		error2:
			sum(k in KS) z[b][k] == n;
	
	forall(j in DSCAT, l in 0..maxlab[j])
		v[<j,l>] <= p[j];
	
	forall(i in IS, j in DSCAT) {
		selcat1:
			v[<j,xcat[i][j]>] + M[j]*f[j] >= 0;
		selcat2:
			v[<j,xcat[i][j]>] - M[j]*f[j] <= 0;
	}

	seltolnum:
		(sum(j in DSCONT, jold in DSCONTOLD) ccont[j][jold]) + (sum(j in DSCAT) f[j]) <= seltol;
}