function predictR = grnmf_fixmodel(args)
 	cd(args.realpath)
    MM_mat=args.drugMat;
 	DD_mat=args.targetMat;
 	MD_mat=args.WR;
    miRNAs_list=string([0:length(MM_mat); 0:length(MM_mat)])';
    diseases_list=string([0:length(DD_mat); 0:length(DD_mat)])';
    inputdata = struct('MD_mat',MD_mat, 'MM_mat',MM_mat,'DD_mat',DD_mat, 'miRNAs_list',miRNAs_list, 'diseases_list', diseases_list);

	params.K=args.K;
    params.r=args.r;
    params.p=args.p;
    params.k = args.k;          
    params.iterate = args.max_iter;   
    params.lamda = args.lambda;         
    params.lamda_m = args.lambda_m;     
    params.lamda_d = args.lambda_d;
    
	[A,B] = GRNMF(inputdata,params);
    predictR = A*B';
end