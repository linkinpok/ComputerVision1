using Images
using PyPlot

include("Common.jl")

#---------------------------------------------------------
# Loads grayscale and color image given PNG filename.
#
# INPUTS:
#   filename     given PNG image file
#
# OUTPUTS:
#   gray         single precision grayscale image
#   rgb          single precision color image
#
#---------------------------------------------------------
function loadimage(filename)
  rgb = PyPlot.imread(filename)*1.0
  gray = Common.rgb2gray(rgb)
  return gray::Array{Float64,2}, rgb::Array{Float64,3}
end


#---------------------------------------------------------
# Computes entries of Hessian matrix.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing image
#   fsize           filter size for smoothing
#
# OUTPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#
#---------------------------------------------------------
function computehessian(img::Array{Float64,2},sigma::Float64,fsize::Int)
  I = imfilter(img, Common.gauss2d(sigma, [fsize fsize]), "replicate")
  deriv = [0.5 0 -0.5]
  I_x = imfilter(I, deriv, "replicate")
  I_y = imfilter(I, deriv', "replicate")
  I_xx = imfilter(I_x, deriv, "replicate")
  I_yy = imfilter(I_y, deriv', "replicate")
  I_xy = imfilter(I_x, deriv', "replicate")
  return I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes function values of Hessian criterion.
#
# INPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#   sigma      std that was used for smoothing image
#
# OUTPUTS:
#   criterion  function score
#
#---------------------------------------------------------
function computecriterion(I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}, sigma::Float64)
  criterion = sigma^4 * (I_xx.*I_yy - I_xy.^2)
  return criterion::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of criterion function values.
#   Extracts local maxima within a 5x5 window and
#   allows multiple points with equal values within the same window.
#   Discards interest points in a 5 pixel boundary.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   criterion  function score
#   thresh     param for thresholding
#
# OUTPUTS:
#   rows        row positions of kept interest points
#   columns     column positions of kept interest points
#
#---------------------------------------------------------
function nonmaxsupp(criterion::Array{Float64,2}, thresh::Float64)
  findMax(X) = X.*(X.==maximum(X))
  #img_filtered = Common.nlfilter(criterion,findMax,5,5,"replicate")
  crit_max = zeros(size(criterion))
  wsize = 5
  # Find maxima in each window
  for r in 1+wsize:wsize:size(crit_max,1)-wsize
    for c in 1+wsize:wsize:size(crit_max,2)-wsize
      crit_max[r:r+wsize-1,c:c+wsize-1] = findMax(criterion[r:r+wsize-1,c:c+wsize-1])
    end
  end
  # Apply threshold
  rows, columns = Common.findnonzero(crit_max .- thresh)
  return rows::Array{Int,1},columns::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Interest point detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = 4.5               # std for presmoothing image
  fsize = 25               # filter size for smoothing
  threshold = 0.001           # Corner criterion threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")

  # Compute the three components of the Hessian matrix
  I_xx,I_yy,I_xy = computehessian(gray,sigma,fsize)

  # Compute Hessian based corner criterion
  criterion = computecriterion(I_xx,I_yy,I_xy,sigma)

  # Display Hessian criterion image
  figure()
  imshow(criterion,"jet",interpolation="none")
  axis("off")
  title("Determinant of Hessian")
  gcf()

  # Threshold corner criterion
  mask = criterion .> threshold
  rows, columns = Common.findnonzero(mask)
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points without non-maximum suppression")
  gcf()

  # Apply non-maximum suppression
  rows,columns = nonmaxsupp(criterion,threshold)

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points after non-maximum suppression")
  gcf()
  return nothing
end
