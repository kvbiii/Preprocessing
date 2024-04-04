%macro create_data(problem_type);
    %let num_rows = 1000;
    %let num_cols = 10;
    data X;
        array x_arr[&num_cols.] x1-x&num_cols.;
        do i = 1 to &num_rows.;
            do j = 1 to &num_cols.;
                x_arr[j] = rand("Uniform", 0, 100);
            end;
			mnozenie=x_arr[4]*x_arr[9];
            output;
        end;
        drop i j;
    run;
	
	proc univariate data=X noprint;
		var mnozenie;
		output out=temp_table pctlpre=P_ pctlpts=50;
	run;
	
	proc sql noprint;
		select * into :percentile trimmed from work.temp_table;
	quit;
	
    data y;
        set X;
        array x_arr[&num_cols.] x1-x&num_cols.;
        if("&problem_type." = "classification") then do;
            if x_arr[4]*x_arr[9] < &percentile then y=1;
			else y = 0;
        end;
        else do;
            y = 0.5*x_arr[4] + 0.5*x_arr[9];
        end;
        drop x1-x&num_cols.;
    run;

    /* merge x and y datasets */
    data dataset;
        merge x y;
		drop mnozenie;
    run;

	proc datasets library=work nolist;
    	delete temp_:;
	run;
%mend;
%create_data(classification);

%macro LOFO(algorithm, dataset, target_name, metric=AUC);
	ods exclude all;	
	proc surveyselect data=&dataset 	
		SAMPRATE=0.7 
		out=select outall 
		method=srs; 
	run;
	ods exclude none;

	data train valid; 
		set select; 
		if selected=1 then 
			output train; 
		else 
			output valid; 
		drop selected;
	run;

	proc sql noprint;
		select name, count(*) into :nazwy_zmiennych SEPARATED BY " ", :liczba_zmiennych trimmed
		from dictionary.columns
		where LIBNAME = upcase("work")
		and MEMNAME = upcase("train")
		and name^="&target_name";
	quit;
	ods exclude all;
	proc logistic data=work.train;    
		model &target_name=&nazwy_zmiennych;
		score data=work.valid out=predictions fitstat;
		ods output ScoreFitStat=work.original_scores;
	run;
	ods exclude none;
	
	proc sql noprint;
		select &metric into :original_score
		from work.original_scores;
	quit;
	%do i=1 %to %sysevalf(&liczba_zmiennych);
		data work.train_copy;
			set work.train;
			drop %scan(&nazwy_zmiennych, &i, ' ');
		run;
		data work.valid_copy;
			set work.valid;
			drop %scan(&nazwy_zmiennych, &i, ' ');
		run;
		proc sql noprint;
			select name into :nazwy_zmiennych_bez_i SEPARATED BY " "
			from dictionary.columns
			where LIBNAME = upcase("work")
			and MEMNAME = upcase("train_copy")
			and name^="&target_name";
		quit;
		ods exclude all;
		proc logistic data=work.train_copy;    
			model &target_name=&nazwy_zmiennych_bez_i;
			score data=work.valid_copy out=predictions fitstat;
			ods output ScoreFitStat=work.valid_scores;
		run;
		ods exclude none;
		proc sql noprint;
			select &metric into :inner_score
			from work.valid_scores;
		quit;
		%let feature_importance&i=%sysevalf(&original_score-&inner_score);
	%end;
	*Feature importance można interpretować jako wielkość utraty mocy predykcyjnej modelu mierzona za pomocą metryki &metric na skutek usunięcia zmiennej &i; 
	data feature_importances;
		length Zmienna $ 24;
		%do i=1 %to %sysevalf(&liczba_zmiennych);
			%let zmienna=%scan(&nazwy_zmiennych, &i, ' ');
			Zmienna="&zmienna";
			Feature_importance=&&feature_importance&i;
			output;
		%end;
	run;
%mend;
%LOFO(algorithm=logistic, dataset=dataset, target_name=y);