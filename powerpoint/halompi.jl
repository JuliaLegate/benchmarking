for it in 1:nt
    diffusion_step!(params, C2, C)
    update_halo!(C2, bufs, neighbors, comm_cart)
    C, C2 = C2, C
end

@views 
function update_halo!(A, bufs, neighbors, comm)
  # dim-1 (x)
  	(neighbors.x[1] != MPI.PROC_NULL) && copyto!(bufs.send_1_1, A[2    , :])
  	(neighbors.x[2] != MPI.PROC_NULL) && copyto!(bufs.send_1_2, A[end-1, :])
  	
  	reqs = MPI.MultiRequest(4)
  	(neighbors.x[1] != MPI.PROC_NULL) && (reqs[1] = MPI.Irecv!(bufs.recv_1_1, neighbors.x[1], comm))
  	(neighbors.x[2] != MPI.PROC_NULL) && (reqs[2] = MPI.Irecv!(bufs.recv_1_2, neighbors.x[2], comm))
  	(neighbors.x[1] != MPI.PROC_NULL) && (reqs[3] = MPI.Isend(bufs.send_1_1, neighbors.x[1], comm))
  	(neighbors.x[2] != MPI.PROC_NULL) && (reqs[4] = MPI.Isend(bufs.send_1_2, neighbors.x[2], comm))
  
  	MPI.Waitall(reqs) # blocking
  	(neighbors.x[1] != MPI.PROC_NULL) && copyto!(A[1 ,  :], bufs.recv_1)
  	(neighbors.x[2] != MPI.PROC_NULL) && copyto!(A[end, :], bufs.recv_1_2)
  
    # dim-2 (y)
  	(neighbors.y[1] != MPI.PROC_NULL) && copyto!(bufs.send_2_1, A[:, 2]) 
  	(neighbors.y[2] != MPI.PROC_NULL) && copyto!(bufs.send_2_2, A[:, end-1])
  
    reqs = MPI.MultiRequest(4)
  	(neighbors.y[1] != MPI.PROC_NULL) && (reqs[1] = MPI.Irecv!(bufs.recv_2_1, neighbors.y[1], comm))
  	(neighbors.y[2] != MPI.PROC_NULL) && (reqs[2] = MPI.Irecv!(bufs.recv_2_2, neighbors.y[2], comm))
  	(neighbors.y[1] != MPI.PROC_NULL) && (reqs[3] = MPI.Isend(bufs.send_2_1, neighbors.y[1], comm))
  	(neighbors.y[2] != MPI.PROC_NULL) && (reqs[4] = MPI.Isend(bufs.send_2_2, neighbors.y[2], comm))
  	
 	MPI.Waitall(reqs) # blocking
  	(neighbors.y[1] != MPI.PROC_NULL) && copyto!(A[:, 1], bufs.recv_2_1) 
	(neighbors.y[2] != MPI.PROC_NULL) && copyto!(A[:, end], bufs.recv_2_2)
	
 	return nothing
 end



