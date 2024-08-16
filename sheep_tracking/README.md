## Technical Details

**Without GPU Parallel Processing:**
29s video : 679 frames : 485s processing time (8 mins 5 sec)

--> 1s video : 23.41 fr : 16.62s processing time

--> 					1 fr		  : 0.714s processing time

if 	  1s video : 16 fr : 11.424s processing time

if    1s video : 10 fr : 7.14s processing time

**With GPU Processing:**
Using an old CPU, 1fr is processed in 0.714s. With a GPU, the processing time for 1 frame would be further reduced. Furthermore, with the parallel processing capability of the GPU, the next 10 frames can be processed in parallel with the first frame, therefore, achiving real-time tracking.