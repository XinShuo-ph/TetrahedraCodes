1. For simno=0, save eigenvales and volumes for all snapshots and for tetsize = 1,4,8,16,32
2. after getting the data, submit other jobs that:
    a. compute density PDFs 
    b. compute sigma at ics
    c. compute density PDFs_ZApts by transporting the eigenvalues
3. plot PDF, PDF_ZApts, theory curves based on sigma*linear growth factor
4. For other simno, don't save eigenvales and volumes, just save the sigma, PDFs and PDFs_ZApts
