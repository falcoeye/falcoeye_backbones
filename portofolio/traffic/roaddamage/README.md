# Models


## Road Damage Models:

A set of models for detecting damages in the road including cracks and potholes


### Model checkpoints 

| Model                  	| Input Image Resolution 	| #params 	| Inf Time (Image/ms)  b=16 	| AP   	| AP50 	| AP75 	| F1    	|
|------------------------	|------------------------	|---------	|:-------------------------:	|------	|------	|------	|-------	|
| [D0 checkpoint](https://drive.google.com/file/d/1fazRz4ZbhUuRMF1UbMAfaTJpNfyhE82U/view?usp=sharing)      	| 512                    	| 3.9M    	| 178                       	| 19.1 	| 47.2 	| 11.5 	| 54.04 	|
| [D1 checkpoint](https://drive.google.com/file/d/1eq8Y-_oEBVlhFyOYlHwIDf83_KEvscAp/view?usp=sharing)      	| 640                    	| 6.5M    	| 147                       	| 21.7 	| 51.5 	| 13.4 	| 56.9  	|
| [D2 checkpoint](https://drive.google.com/file/d/1_FIy_a3EgY7oGtdmpCZGWecMchU-eVC4/view?usp=sharing)       	| 768                    	| 8M      	| 100                       	| 22.9 	| 53.5 	| 14.9 	| 56.7  	|
| [D3 checkpoint](https://drive.google.com/file/d/15Sk7Z5J_jYj7cm7Jar2H7U6h2etn4C_q/view?usp=sharing)       	| 796                    	| 11.9M   	| 57                        	| 23.0 	| 53.4 	| 15.0 	| 56.5  	|
| [D4 checkpoint](https://drive.google.com/file/d/1Q3HQBn986n2ifFx3nR3Oe-bqJ6eugBhS/view?usp=sharing)      	| 1024                   	| 20.5M   	| 38                        	| 22.8 	| 53.3 	| 15.1 	| 57.2  	|


You need to change the handler file and pass the appropriate arguments to the archive script depending on the checkpoint
