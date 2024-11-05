/*********************************************
 * OPL 22.1.1.0 Model
 * Author: songkomkrit
 * Creation Date: Nov 4, 2024 at 12:24:05 AM
 *********************************************/
 
/*********************************************
 * NOTES
 * pl.bc.solutionValue[thisOplModel.mPairs.find(1,0)]
 *********************************************/

/*********************************************
 * Class Labels
 * Input file: 0, 1, 2, ..., n
 * Algorithm: 0, 1, 2, ..., n
 * Output file: 0, 1, 2, ..., n
 *********************************************/

/*********************************************
 * INPUTS
 *********************************************/ 
int mdimold = 3;	// dimension			// 4 or 184 or 8 or 4
int mdimcontold = 1; // continuous dimension	// 2 or 66 or 3 or 2
//int mdimcat = 2; // categorical dimension	// 2 or 118 or 5 or 2
int mN = 100;	// number of instances	// 8 or 157681 or 100 or 100
int mn = 4;		// the value of n = (number of classes) - 1		// 1 or 4 or 4

int mseltol = 2;	// given number of total selected cont/cat dimensions (at most)

// Initialized UB on number of selected continuous dimensions
int mselcont = mdimcontold;
execute {
	if (mselcont > mseltol)
		mselcont = mseltol;
}

int mexccont = mdimcontold - mselcont;	// computed LB on number of excluded continuous dimensions
int mdim = mdimold - mexccont;
int mdimcont = mselcont;

range mDS = 1..mdim;
range mDSCONTOLD = 1..mdimcontold;	// old continuous
range mDSCONT = 1..mselcont;	// new continuous
range mDSCAT = mdimcont+1..mdim;	// shifted categorical
range mIS = 1..mN;
float mxcontold[mIS][mDSCONTOLD];	// x along continuous dimensions
int mxcat[mIS][mDSCAT]; // x along categorical dimensions
int my[mIS];
int mmaxlab[mDSCAT];	// maximum labels for categorical dimensions
float mM[mDS];	// big-M for all new/shifted dimensions (continuous and categorical)
float mm[mDSCONT];	// small-m for continuous dimensions
int mp[mDS];	// number of cuts along axes
int mcoef[mDS];

/*********************************************
 * TUPLES
 *********************************************/ 
tuple ContPairType {	// index for continuous cut
	int j;
	int q;
};

{ContPairType} mContPairs = {<j, q> | j in mDSCONT, q in 0..mp[j]+1};

tuple ContTripleType {	// index for continuous cut of each individual instance
	int i;
	int j;
	int q;
};

{ContTripleType} mContTriples = {<i, j, q> | i in mIS, j in mDSCONT, q in 0..mp[j]};

tuple CatPairType {	// index for categorical group
	int j;
	int l;
};

{CatPairType} mCatPairs = {<j, l> | j in mDSCAT, l in 0..mmaxlab[j]};

tuple tuplePred {
	key int b;
	sorted {int} label;
}
sorted {tuplePred} mpred;
{int} memptyset = {};

/*********************************************
 * OUTSIDE EXECUTION
 *********************************************/
execute {
	thisOplModel.settings.run_engineLog = "tmp/current-engine.log";	// temporary engine log
}

/*********************************************
 * MAIN EXECUTION
 *********************************************/ 
main {
	var ftime = Opl.round((new Date()).getTime()/1000) % 100000; // first timestamp (in seconds)
	
	// Input/variable filenames
	var infilename = "input/seltrain20num3each20.csv";			// input filename
	var varfilename = "input/selproc20num3co3ca3cutinfo.csv";	// variable filename (6 columns)
	
	// Prefix of all output files
	var prefixout = "output/" + ftime + "-";
	prefixout += infilename.split("/")[1].split(".")[0] + "-";

	// Inputs
	//var M0 = 500;			// big-M (float)
	var m0 = 0.01;			// small-m (float)
	var pcont0 = 3;			// max number of cuts along continuous axis (integer)
	
	// Customization
	var timelimit = 1;	// whether set total time limits (1 = limit / 0 = none)
	var limit = 1;		// whether customize performance settings (1 = customize / 0 = none)
	var perf = 1;		// whether set limits (1 = limit / 0 = none)
	
	// Custom time limit parameter
	if (timelimit == 1)
		var acctimelimmin = 24*60;	// accumulated time limit (in minutes)
	
	// Cplex limit parameters (excluding time limit)
	if (limit == 1) {
		var intsollim = 1;	// MIP solution number limit (in each iteration)
	}	 
	
	// Cplex performance parameters
	if (perf == 1) {
		var threads = 0;	// parallel threads (default: 0 = at most 32 threads)
		var workmemgb = 2;	// working memory before compression and swap (in GB) (default: 2 GB) (only marginally improved efficiency)	
		var trelimgb = 200;	// uncompressed tree memory limit (in GB) (default: around 1e+72 GB)
		
		/* Node storage file switch
		 * 0 = No node file
		 * 1 = Node file in memory and compressed (default)
		 * 2 = Node file on disk
		 * 3 = Node file on disk and compressed
		 */
		var nodefileind = 3;
		
		/* Note on directory for temporary working files
		 * cplex.workdir = ...;
		 * CPLEX Error 1422: Could not open file for writing
		 */
		
		// Calculation
		var workmem = 1024*workmemgb;	// working memory before compression and swap (in MB) (default: 2048 MB)
		var trelim = 1024*trelimgb;		// uncompressed tree memory limit (in MB) (default: 1e+75 MB)
	}
	
	// Postfixes
	var cpostfixname = "mfullaltseltol-" + thisOplModel.mseltol;	// common postfix name
	if (timelimit == 1)
		cpostfixname += "-t-" + acctimelimmin + ".csv";
	else
		cpostfixname += ".csv";
	var postfixerror = "-" + cpostfixname;	// postfix of error file
	var postfixout = "-pcont-" + pcont0 + "-" + cpostfixname;	// postfix of all other output files
	
	// Output filenames
	var outerrorname = prefixout + "export-error" + postfixerror;
	var outinstancename = prefixout + "export-predict-instance" + postfixout;
	var outcutcontname = prefixout + "export-cutcont-full" + postfixout;
	var outcutcatname = prefixout + "export-cutcat-full" + postfixout;
	// The existence of region is not checked here
	// In fact, it can be check through enumeration of certain binary representations
	var outregionname = prefixout + "export-predict-region" + postfixout;
	var outselvarintname = prefixout + "export-select-var-int" + postfixout; // selected variables (integer)
	var outselvarstrname = prefixout + "export-select-var-str" + postfixout; // selected variables (string)
	
	// Engine log (initialized)
	var logfilename = "log/" + ftime + "-engine-" + cpostfixname.split(".")[0] + ".log";
	var outlog = new IloOplOutputFile(logfilename);
	
	// OPL
	var source = new IloOplModelSource("p-mixed-cuts-alt-seltol.mod");
	var cplex = new IloCplex();
	var def = new IloOplModelDefinition(source);
	var opl = new IloOplModel(def,cplex);
	var data = new IloOplDataElements();
	
	data.dimold = thisOplModel.mdimold;
	data.dimcontold = thisOplModel.mdimcontold;
	data.dim = thisOplModel.mdim;
	data.dimcont = thisOplModel.mdimcont;
	//data.dimcat = thisOplModel.mdimcat;
	data.N = thisOplModel.mN;
	data.n = thisOplModel.mn;
	data.xcontold = thisOplModel.mxcontold;
	data.xcat = thisOplModel.mxcat;
	data.y = thisOplModel.my;

	var pred = thisOplModel.mpred;		// set of predicted labels
	
	data.seltol = thisOplModel.mseltol;
	data.selcont = thisOplModel.mselcont;
	data.exccont = thisOplModel.mexccont;
	
	data.m = thisOplModel.mm;
	for (var j=1; j<=data.dimcont; j++)
		data.m[j] = m0;
	
	var f = new IloOplInputFile(infilename);	// training dataset
	f.readline();								// skip a header
	for (var i=1; i<=data.N; i++) {
		var myitem = f.readline().split(",");
		data.y[i] = Opl.intValue(myitem[data.dimold]);
		for (var j=1; j<=data.dimcontold; j++)
			data.xcontold[i][j] = Opl.floatValue(myitem[j-1]);
		for (var j=data.dimcontold+1; j<=data.dimold; j++)
			data.xcat[i][j-data.exccont] = Opl.intValue(myitem[j-1]);
	}
	f.close();

	data.p = thisOplModel.mp;
	for (var j=1; j<=data.dimcont; j++)
		data.p[j] = pcont0;
	
	data.M = thisOplModel.mM;
	data.maxlab = thisOplModel.mmaxlab;
	var M0cont = 1;
	var f = new IloOplInputFile(varfilename);	// variable info
	f.readline();								// skip a header
	for (var j=1; j<=data.dimold; j++) {
		var myitem = f.readline().split(",");
		if (j <= data.dimcontold) {
			var curMcont = 1 + Opl.maxl(Opl.abs(Opl.intValue(myitem[3])), Opl.abs(Opl.intValue(myitem[4])));
			M0cont = Opl.maxl(M0cont, curMcont);
		}		
		else {
			data.p[j-data.exccont] = Opl.intValue(myitem[5]);
			data.maxlab[j-data.exccont] = Opl.intValue(myitem[4]);
			data.M[j-data.exccont] = 1 + Opl.intValue(myitem[5]);
 		}			
	}
	f.close();
	
	for (var j=1; j<=data.dimcont; j++)
		data.M[j] = M0cont;
	
	data.coef = thisOplModel.mcoef;	
	data.coef[1] = 1;
	for (var j=2; j<=data.dim; j++)
		data.coef[j] = data.coef[j-1]*(data.p[j]+1);
	
	var nump = 0;		// total number of cuts
	for (var j=1; j<=data.dim; j++)
		nump += data.p[j];
	
	opl.addDataSource(data);
	opl.generate();
	opl.settings.mainEndEnabled = true;
	
	// Cplex limits (excluding time limit)
	if (limit == 1) {
		cplex.intsollim = intsollim;	// MIP solution number limit (> 0)
	}
	
	// Cplex performance
	if (perf == 1) {
		cplex.threads = threads;	// parallel threads
		cplex.workmem = workmem;	// working memory before compression and swap (in MB)
		cplex.trelim = trelim;		// uncompressed tree memory limit (in MB)
		cplex.nodefileind = nodefileind;	// node storage file switch
	}
	
	// Initialization
	var status = -9;	// solution status code (initialized)
	var iter = 0;		// iteration
	var acctime = 0;	// accumulated running time (in seconds)
	var texceed = 0;	// whether acctime > tilimmin (1 = total time limit exceeded / 0 = not)
	
	// Calculation
	if (timelimit == 1)
		var acctimelim = 60*acctimelimmin;	// accumulated time limit (in seconds)
	else
		var acctimelim = -1;
	
	// Optimization
	while (texceed == 0) {	// accumulated time limit not exceeded
	
		// Exit status codes
		if (status == 1)		// 1: CPX_STAT_OPTIMAL
			break;
		else if (status == 101)	// 101: CPXMIP_OPTIMAL
			break;
		else if (status == 102)	// 102: CPXMIP_OPTIMAL_TOL
			break;
		else if (status == 111)	// 111: CPXMIP_MEM_LIM_FEAS
			break;
		else if (status == 112)	// 112: CPXMIP_MEM_LIM_INFEAS
			break;
		
		/* Non-exit status codes
		 * 11: CPX_STAT_ABORT_TIME_LIM
		 * 104: CPXMIP_SOL_LIM
		 */
		
		// In the case when the previous status is not one of the above
		if (timelimit == 1)			// time limit for each call to optimizer (in seconds)
			cplex.tilim = acctimelim - acctime;
		var start = new Date();		// begin a timer
		
		pred.clear();	// clear previous set of predicted labels
		
		// Solve
		if (cplex.solve()) {
		
			var end = new Date();	// end a timer
			var solvetime = end.getTime() - start.getTime();	// compute solving time
			acctime += solvetime/1000;	// accumulated running time (in s)
			
			if ((timelimit == 1) && (acctime >= acctimelim))	// total time limit exceeded (in seconds)
				texceed = 1;
			
			iter += 1;	// update iteration
			
			var error = data.N + cplex.getObjValue();	// the number of misclassified instances
			var accuracy = (1-error/data.N)*100;	// training accuracy
			
			status = cplex.status;	// solution status code (1 = opt / 11 = time limit / ...)
			var lberr = data.N + cplex.getBestObjValue();	// LB on minimum (optimal) error
			var relgap = cplex.getMIPRelativeGap();	// relative objective gap for MIP
			
			// Open output text files (append = true)
			var outerror = new IloOplOutputFile(outerrorname, true);
			var outinstance = new IloOplOutputFile(outinstancename, true);
			var outcutcont = new IloOplOutputFile(outcutcontname, true);
			var outcutcat = new IloOplOutputFile(outcutcatname, true);
			var outregion = new IloOplOutputFile(outregionname, true);
			var outselvarint = new IloOplOutputFile(outselvarintname, true);
			var outselvarstr = new IloOplOutputFile(outselvarstrname, true);
			
			// outerror
			if (!outerror.exists) {
				outerror.write("iter,");
				for (var j=1; j<=data.dim; j++)	
					outerror.write("p", j, ",");
				outerror.write("error,accuracy,ms,acctmin,status,lberr,relgap");
			}		
			outerror.write("\n", iter, ",");
			for (var j=1; j<=data.dim; j++)
				outerror.write(data.p[j], ",");		
			outerror.write(error, ",", accuracy, ",");
			outerror.write(solvetime, ",", acctime/60, ",");
			outerror.write(status, ",", lberr, ",", relgap);
	
			// Scripting logs 1
			writeln("\n------------------------------");
			writeln("Iteration ", iter);
			writeln("Bounds on # of cuts = ", nump, " with", data.p);
			writeln("Error = ", error, " (out of ", data.N, " instances)");
			writeln("Accuracy = ", accuracy);
			writeln("Solving time = ", solvetime/60000, " min (minutes)");
			writeln("Accumulated time = ", acctime/60, " min (minutes)");
			writeln("\nSolution status code = ", status);
			writeln("LB on error =  ", lberr);
			writeln("Relative objective gap = ", relgap);
			writeln("\nSelected variables:");
	
			// Create a set of predicted labels (majority voting)
			for (var b=0; b<opl.B; b++) {
				var lset = Opl.operatorUNION(thisOplModel.memptyset,thisOplModel.memptyset);				
				var maxnum = 0;
				for (var k=0; k<=data.n; k++) {
					var num = 0;
					for (var i=1; i<=data.N; i++)
						num += (data.y[i] == k)*opl.g.solutionValue[i][b];						
					if (num == maxnum)
						lset.add(k);						
					else if (num > maxnum) {
						maxnum = num;						
						lset.clear();
						lset.add(k);						
					}				
				}
				pred.add(b, lset);				
			}
			
			// outinstance
			if (!outinstance.exists)
				outinstance.write("iter,id,class,region,predict");
			for (var i=1; i<=data.N; i++) {
				outinstance.write("\n", iter, ",", i, ",", data.y[i], ",");
				for (var b=0; b<opl.B; b++) 
					if (opl.g.solutionValue[i][b] == 1) {	// occur only once
						outinstance.write(b, ",");
						outinstance.write(pred.get(b).label);	
						break;	// terminate the loop
					}		
			}
			
			// outcutcont
			if (!outcutcont.exists)
				outcutcont.write("iter,j,q,bc");
			for (var j=1; j<=data.dimcont; j++) {
				for (var q=1; q<=data.p[j]; q++) {
					outcutcont.write("\n", iter, ",", j, ",", q, ",");
					outcutcont.write(opl.bc.solutionValue[thisOplModel.mContPairs.find(j,q)]);		
				}
			}
	
			// outcutcat
			if (!outcutcat.exists)
				outcutcat.write("iter,j,l,v");
			for (var j=data.dimcont+1; j<=data.dim; j++) {
				for (var l=0; l<=data.maxlab[j]; l++) {
					outcutcat.write("\n", iter, ",", j, ",", l, ",");
					outcutcat.write(opl.v.solutionValue[thisOplModel.mCatPairs.find(j,l)]);		
				}	
			}
			
			// outregion
			if (!outregion.exists)
				outregion.write("iter,region,occupy,predict");
			for (var b=0; b<opl.B; b++) {
				outregion.write("\n", iter, ",", b, ",");
				var s = 0;		// initialize s (presumably unoccupied)
				for (var i=1; i<=data.N; i++)
					if (opl.g.solutionValue[i][b] == 1) {	// occupied
						s = 1;
						break;	// iterminate the loop					
					}
				outregion.write(s, ",");
				outregion.write(pred.get(b).label);		
			}
			
			// outselvarint
			if (!outselvarint.exists)
				outselvarint.write("iter,j,jold,mselect,type");	// mselect = model select (not actual)
			for (var j=1; j<=data.dimcont; j++) {	// selected continuous features
				outselvarint.write("\n", iter, ",", j, ",");
				var seljold = -1;
				for (var jold=1; jold<=data.dimcontold; jold++)
					// Determine which old continuous feature is selected
					if (opl.ccont.solutionValue[j][jold] == 1) {
						seljold = jold;
						break;	// terminate the loop
					}
				outselvarint.write(seljold, ",");
				outselvarint.write("1,");	// Based on model, all new cont features are selected
				outselvarint.write("cont");	
			}
			for (var j=data.dimcont+1; j<=data.dim; j++) {	// categorical feature
				outselvarint.write("\n", iter, ",", j, ",", j+data.exccont, ",");
				if (opl.f.solutionValue[j] == 1)	// selected categorical feature (model)
					outselvarint.write("1,");
				else	// unselected categorical feature (model)
					outselvarint.write("0,");
				outselvarint.write("cat");	
			}
			
			// outselvarstr
			if (!outselvarstr.exists)
				outselvarstr.write("iter,jold,jnew,aselect,type,variable"); // aselect = actual select
			var varinfile = new IloOplInputFile(varfilename);		// variable info
			varinfile.readline();	// skip a header
			var numselcont = 0;	// initialized number of actually selected continuous features
			var numselcat = 0;	// initialized number of actually selected categorical features
			for (var jold=1; jold<=data.dimcontold; jold++) {	// CONTINUOUS
			 	outselvarstr.write("\n", iter, ",", jold, ",");
			 	var jnew = -1;
			 	var aselect = 0;	// initialized to be unselected (continuous)
			 	for (var j=1; j<=data.dimcont; j++)
			 		// Determine whether a current old continuous feature is selected
			 		if (opl.ccont.solutionValue[j][jold] == 1) {	// selected (actual 1/2)
			 			jnew = j;
			 			break;	// terminate the loop
			 		}
			 	outselvarstr.write(jnew, ",");
			 	var myitem = varinfile.readline().split(",");
			 	if (jnew > 0) {	// selected continuous feature (actual 1/2)
			 		aselect = 1;	// seem to be selected (initialization for actual 2/2)
			 		for (var q=0; q<=data.p[jnew]; q++) {
			 			var bcleft = opl.bc.solutionValue[thisOplModel.mContPairs.find(jnew,q)];
			 			var bcright = opl.bc.solutionValue[thisOplModel.mContPairs.find(jnew,q+1)];
			 			var minxjnew = Opl.intValue(myitem[3]);
			 			var maxxjnew = Opl.intValue(myitem[4]);
			 			if ((bcleft <= minxjnew) && (bcright >= maxxjnew)) {	// cover [min,max]
			 				aselect = 0;	// unselected (actual 2/2)
			 				break;
			 			}
			 		}
	  			}		 	
			 	outselvarstr.write(aselect, ",");
			 	if (aselect == 1) { // actually selected continuous feature
			 		// Scripting logs 2 (continuous)
			 		write("\t", myitem[1], " (Continuous)\n");
			 		numselcont += 1;	 	  
			 	}
			 	outselvarstr.write("cont,");
			 	outselvarstr.write(myitem[1]);	// variable name
			}
			for (var jold=data.dimcontold+1; jold<=data.dimold; jold++) {	// CATEGORICAL
				var jnew = jold-data.exccont;
				outselvarstr.write("\n", iter, ",", jold, ",", jnew, ",");
				var aselect = 0;	// initialized to be unselected (categorical)
				var myitem = varinfile.readline().split(",");
				if (opl.f.solutionValue[jnew] == 1) {	// selected categorical feature (actual 1/2)
	 				var vat0 = opl.v.solutionValue[thisOplModel.mCatPairs.find(jnew,0)];
	 				for (var l=1; l<=data.maxlab[jnew]; l++) {
	 					var vcur = opl.v.solutionValue[thisOplModel.mCatPairs.find(jnew,l)];
	 					if (vcur != vat0) {	// distinct new groups are detected
	 						aselect = 1;	// selected categorical feature (actual 2/2)
	 						break;
	 					}
	 				}
	 			}
	 			outselvarstr.write(aselect, ",");
	 			if (aselect == 1) {	// actually selected categorical feature
					// Scripting logs 2 (categorical)
					write("\t", myitem[1], " (Categorical)\n");
					numselcat += 1;				
	 			}
				outselvarstr.write("cat,");	
			 	outselvarstr.write(myitem[1]);
			}
			varinfile.close();
			
			// Scripting logs 3
			var numselall = numselcont + numselcat;
			writeln("\nNumber of selected variables = ", numselall, " (", numselcont, " continuous + ", numselcat, " categorical)");
			writeln("------------------------------");
			
			// Closing output text files
			outerror.close();
			outinstance.close();
			outcutcont.close();
			outcutcat.close();
			outregion.close();
			outselvarint.close();
			outselvarstr.close();
		}
		else
			writeln("No solution");
	}		
	
	opl.end();
	data.end(); 
	def.end(); 
	cplex.end(); 
	source.end();
	
	// Engine log (exported)
	var inlog = new IloOplInputFile("tmp/current-engine.log");
	while (!inlog.eof) {
		outlog.writeln(inlog.readline());
	}
	inlog.close();
	outlog.close();
}
