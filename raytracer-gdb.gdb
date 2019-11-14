# -> Vector operations

define createVector
  set $arg0 = (float*)malloc(($argc-1)*sizeof(float))
  set $i = 1
  while $i < $argc
    eval "set $arg0[$i-1]=$arg%d", $i
    set $i = $i+1
  end
end

define dotProd
  set $i = 0
  set $arg0 = 0
  while $i < 3
    set $arg0 = $arg0 + $arg1[$i]*$arg2[$i]
    set $i = $i + 1
  end
end

define magVector
  set $arg0 = 0
  set $i = 0
  while $i < 3
    set $arg0 = $arg0+__pow($arg1[$i],2)
    set $i = $i+1
  end
  set $arg0 = __sqrt($arg0)
end

define normVector
  magVector $temp $arg0
  set $arg0[0] = $arg0[0]/$temp
  set $arg0[1] = $arg0[1]/$temp
  set $arg0[2] = $arg0[2]/$temp
end

define subVector
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $i = 0
  while $i < 3
    set $arg0[$i] = $arg1[$i]-$arg2[$i]
    set $i = $i+1
  end
end

define sumVector
  set $i = 0
  while $i < 3
    set $arg0[$i] = $arg0[$i]+$arg1[$i]
    set $i = $i+1
  end
end

define scaleVector
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $i = 0
  while $i < 3
    set $arg0[$i] = $arg1[$i]*$arg2
    set $i = $i+1
  end
end

define crossVectors
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $i = 1
  set $k = 0
  while $k < 3
    set $temp = ($i+1)%3
    set $arg0[$k]=$arg1[$i]*$arg2[$temp]-$arg1[$temp]*$arg2[$i]
    set $i = $temp
    set $k = $k+1
  end
end 

define copyVector
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $i = 0
  while $i < 3
    set $arg0[$i] = $arg1[$i]
    set $i = $i+1
  end
end

define negVector
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $arg0[0] = -$arg1[0]
  set $arg0[1] = -$arg1[1]
  set $arg0[2] = -$arg1[2]
end

define linComb
  scaleVector $arg0 $u $arg1[0]
  scaleVector $temp1 $v $arg1[1]
  scaleVector $temp2 $w $arg1[2]
  sumVector $arg0 $temp1
  sumVector $arg0 $temp2
end

# arg0:rl, arg1:d, arg2:n
define reflect
  copyVector $arg0 $arg1
  dotProd $tempAlig $arg1 $arg2
  scaleVector $nSca $arg2 -2*$tempAlig
  sumVector $arg0 $nSca
  normVector $arg0
end

# arg0:rr, arg1:d, arg2:n, arg3:IRo/IRi
define refract
  dotProd $tempAlig $arg1 $arg2
  if $tempAlig > 0
    negVector $tempNorm $arg2
    dotProd $tempAlig $arg1 $tempNorm
  end
  if $tempAlig <= 0
    set $tempNorm = $arg2
  end

  copyVector $semiRefl $arg1
  scaleVector $nScaAlig $tempNorm -$tempAlig
  sumVector $semiRefl $nScaAlig
  scaleVector $semiRflSca $semiRefl $arg3
  printVector3 $semiRflSca

  set $tempRfrAlig = __sqrt(1-(1-$tempAlig*$tempAlig)*$arg3*$arg3)
  scaleVector $nScaRfrAlig $tempNorm $tempRfrAlig
  printVector3 $nScaRfrAlig

  subVector $arg0 $semiRflSca $nScaRfrAlig
  normVector $arg0
end

define printVector3
  p {$arg0[0], $arg0[1], $arg0[2]}
end

define printVector2
  p {$arg0[0], $arg0[1]}
end
  

# Plane operations

define scaleOverPlane
  set $temp_depth = $arg0[2]
  if $arg0[2] < 0
    set $temp_depth = -$arg0[2]
  end
  set $arg0[0] = $arg0[0]*(0.1/$temp_depth)
  set $arg0[1] = $arg0[1]*(0.1/$temp_depth)
  set $arg0[2] = $arg0[2]*(0.1/$temp_depth)
end

# Deprecated
define projectOverPlane
  set $arg0 = (float*)malloc(3*sizeof(float))
  negVector $temp $w
  dotProd $arg0[0] $arg1 $u
  dotProd $arg0[1] $arg1 $v
  dotProd $arg0[2] $arg1 $temp
end

define getWCoords
  set $arg0 = (int*)malloc(2*sizeof(int))
  set $arg0[0] = $width*($arg1[0]-$left)/($right-$left)-0.5
  set $arg0[1] = $height*($arg1[1]-$bottom)/($top-$bottom)-0.5
end

define getCamCoords
  set $arg0 = (float*)malloc(2*sizeof(float))
  set $arg0[0] = $left+(($right-$left)/$width)*($arg1[0]+0.5)
  set $arg0[1] = $bottom+(($top-$bottom)/$height)*($arg1[1]+0.5)
end


# -> Matrix operations

define createMatrix
  set $arg0 = (float*)malloc(9*sizeof(float))
  set $i = 0
  set $j = 0
  while $i < 9
    set $arg0[$i+0] = $arg1[$j]
    set $arg0[$i+1] = $arg2[$j]
    set $arg0[$i+2] = $arg3[$j]
    set $i = $i+3
    set $j = $j+1
  end
end

define matrixVecProd
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $i = 0
  set $j = 0
  set $k = 0
  while $i < 9
    while $j < 3
      set $arg0[$k] = $arg0[$k]+$arg1[$i+$j]*$arg2[$j]
      set $j = $j+1
    end
    set $i = $i+3
    set $j = 0
    set $k = $k+1
  end
end


# -> Sphere operations

define createSphere
  set $arg0 = (float*)malloc(3*sizeof(float))
  set $arg0[0] = $arg2
  set $arg0[1] = $arg3
  set $arg0[2] = $arg4
  set $arg1 = (float*)malloc(3*sizeof(float))
  set $arg1[0] = $arg5
  set $arg2[1] = $arg6
  set $arg3[2] = $arg7
end

define sphereNormal
  subVector $arg0 $arg1 $arg2
  normVector $arg0
end

# arg0:t, arg1:p, arg2:e arg3:d, arg4:c, arg5:r
define sphereInter
  subVector $tempEC $arg2 $arg4
  dotProd $tempB $tempEC $arg3
  set $tempB = 2*$tempB
  dotProd $tempEC2 $tempEC $tempEC
  set $tempC = $tempEC2-$arg5*$arg5
  set $tempDET = $tempB*$tempB - 4*$tempC
  if $tempDET >= 0
    set $tempDET = __sqrt($tempDET)
    set $tempT2 = -$tempB+$tempDET
    set $tempT1 = -$tempB-$tempDET
    if $tempT2 >= 0
      if $tempT1>0 and $tempT1<2*$arg0
        set $tempT1 = $tempT1/2
        set $arg0 = $tempT1 
      end
      if $tempT1<0 and $tempT2<2*$arg0
        set $tempT2 = $tempT2/2
        set $arg0 = $tempT2
      end
      copyVector $arg1 $arg2
      scaleVector $tempRay $arg3 $arg0
      sumVector $arg1 $tempRay
    end
  end
end


# -> Experiments


b main
r -s scene11.json -w 128 -h 128


# Set camera coords and find directions

createVector $e 0 0 4
createVector $target 0 0 0
createVector $up 0 1 0
subVector $w $e $target
normVector $w
crossVectors $u $up $w
normVector $u
crossVectors $v $w $u
set $top = 0.041421 
set $bottom = -$top 
set $right = 0.041421 
set $left = -$right
set $width = 128
set $height = 128


# Choose objective

createVector $ij 69 43
getCamCoords $icjc $ij
createVector $coords $icjc[0] $icjc[1] -0.1
linComb $d $coords
normVector $d



# Trace Ray!

b 955 if i==$ij[0] and j==$ij[1]
c
b 546
c
#c
#b 504
#c
#b 896
#c
#b 605
#c
#p e
#copyVector $newObjective $e
#scaleVector $toObjective d $tObjective
#sumVector $newObjective $toObjective
#printVector3 $newObjective
#b Sphere::intersectsRay if _r==0.4
#c


print "Success!!"

