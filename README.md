# ML1-Final-Project

### ***What is the Golden Ratio?***
                                     
                                     
Aesthetics and appearance play a major role in our society today. While there is no true definition of beauty, our ability to find reliable patterns and proportions that extend into art, aesthetics, and ideals of beauty are seen everywhere today. The golden ratio is one of many theories of what makes an object perceived as pleasing. 

                                          
The golden ratio is rooted in a mathematical proportion, the Greek letter φ (phi), which is viewed as the ideal proportion. The proportion mathematically works out that one proportion is related to the other by 1.618. This number is derived from the Fibonacci sequence and other instances in nature. 
                                     

### ***Our Study***
                                     

In the nature of facial aesthetics, the golden ratio is one of many frameworks for perceived attractiveness through balance and harmony in features. Although there are many different ratios that can be viewed and analyzed, in our study, we focused on measurements between face length to width, eye spacing, and nose to mouth ratio. This is only a subset of the proportions deemed as “ideal”. 
         

### ***Our Data***
                                     

We created our own dataset using data from our cohort. We collected information such as name, email, race, gender, and a headshot photo from 52 individuals within our cohort. We then used a convolutional neural network including python packages from opencv and mediapipe to build a complete dataset featuring measurements such as mouth width, nose width, face width, and face height. These features were then used to create face and mouth-nose ratios and averaged to create an overall “golden ratio score” or proportionality score. This complete dataframe is used for all of our models. 

                           
### ***Complications and Caveats***
                                     

Because of inconsistency in headshot files, posture, and quality of images, not all measurements from all individuals were able to be taken. In addition, small changes in posture and quality could affect our CNNs ability to correctly detect facial features. facial feature detection trial and error with using different networks was frustrating to say the least. OpenCV facial width and height originated as a square, eyebrows would be recognized as noses, background noise from images would be detected as faces themselves, or nothing would be detected at all. We went through over three different CNNs before settling on one and being satisfied that most faces were detected accurately. Moreover, we had a very small sample size, which most likely indicates that our analysis is not an accurate representation of the cohort. 

              
