using Images
using PyPlot
using Printf
using Random
using Statistics
using LinearAlgebra
using Interpolations

include("Common.jl")


#---------------------------------------------------------
# Loads keypoints from JLD2 container.
#
# INPUTS:
#   filename     JLD2 container filename
#
# OUTPUTS:
#   keypoints1   [n x 2] keypoint locations (of left image)
#   keypoints2   [n x 2] keypoint locations (of right image)
#
#---------------------------------------------------------
function loadkeypoints(filename::String)
  keypoints1 = load(filename,"keypoints1")
  keypoints2 = load(filename,"keypoints2")
  @assert size(keypoints1,2) == 2
  @assert size(keypoints2,2) == 2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end


#---------------------------------------------------------
# Compute pairwise Euclidean square distance for all pairs.
#
# INPUTS:
#   features1     [128 x m] descriptors of first image
#   features2     [128 x n] descriptors of second image
#
# OUTPUTS:
#   D             [m x n] distance matrix
#
#---------------------------------------------------------
function euclideansquaredist(features1::Array{Float64,2},features2::Array{Float64,2})
  D = zeros(size(features1,2),size(features2,2))
  for r in 1:size(features1,2)
    for c in 1:size(features2,2)
      D[r,c] = sum((features1[:,r]-features2[:,c]).^2)
    end
  end
  @assert size(D) == (size(features1,2),size(features2,2))
  return D::Array{Float64,2}
end


#---------------------------------------------------------
# Find pairs of corresponding interest points given the
# distance matrix.
#
# INPUTS:
#   p1      [m x 2] keypoint coordinates in first image.
#   p2      [n x 2] keypoint coordinates in second image.
#   D       [m x n] distance matrix
#
# OUTPUTS:
#   pairs   [min(N,M) x 4] vector s.t. each row holds
#           the coordinates of an interest point in p1 and p2.
#
#---------------------------------------------------------
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
  m = size(p1,1)
  n = size(p2,1)
  #pairs = zeros(min(m,n),4)
  pairs = Array{Int,2}(undef, min(m,n),4)
  if m <= n
    pairs[:,1:2] = p1
    for r = 1:m
      mindist, corr = findmin(D[r,:])
      pairs[r,3:4] = p2[corr,:]
    end
  else
    pairs[:,3:4] = p2
    for c = 1:n
      mindist, corr = findmin(D[:,c])
      pairs[c,1:2] = p1[corr,:]
    end
  end
  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end


#---------------------------------------------------------
# Show given matches on top of the images in a single figure.
# Concatenate the images into a single array.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   pairs   [n x 4] vector of coordinates containing the
#           matching pairs.
#
#---------------------------------------------------------
function showmatches(im1::Array{Float64,2},im2::Array{Float64,2},pairs::Array{Int,2})
  figure()
  imshow(hcat(im1,im2),"gray")
  plot(pairs[:,1],pairs[:,2],"xy")
  plot(pairs[:,3].+size(im1,2),pairs[:,4],"xy")
  for i in 1:size(pairs,1)
    plot([pairs[i,1], pairs[i,3]+size(im1,2)],pairs[i,2:2:4])
  end
  return nothing::Nothing
end


#---------------------------------------------------------
# Computes the required number of iterations for RANSAC.
#
# INPUTS:
#   p    probability that any given correspondence is valid
#   k    number of samples drawn per iteration
#   z    total probability of success after all iterations
#
# OUTPUTS:
#   n   minimum number of required iterations
#
#---------------------------------------------------------
function computeransaciterations(p::Float64,k::Int,z::Float64)
  #z = 0.99
  n = log(1-z)/log(1-p^k)
  n = Int(ceil(n))
  return n::Int
end


#---------------------------------------------------------
# Randomly select k corresponding point pairs.
#
# INPUTS:
#   points1    given points in first image
#   points2    given points in second image
#   k          number of pairs to select
#
# OUTPUTS:
#   sample1    selected [kx2] pair in left image
#   sample2    selected [kx2] pair in right image
#
#---------------------------------------------------------
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
  r = rand(1:size(points1,1),k)
  sample1 = points1[r,:]
  sample2 = points2[r,:]
  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end


#---------------------------------------------------------
# Conditioning: Normalization of coordinates for numeric stability.
#
# INPUTS:
#   points    unnormalized coordinates
#
# OUTPUTS:
#   U         normalized (conditioned) coordinates
#   T         [3x3] transformation matrix that is used for
#                   conditioning
#
#---------------------------------------------------------
function condition(points::Array{Float64,2})
  points_c = Common.hom2cart(points)
  s = 0.5 * maximum(norm.(points_c))
  t = [mean(points_c[1,:]); mean(points_c[2,:])]
  T = [1/s 0 -t[1]/s;
       0 1/s -t[2]/s;
       0 0 1]
  U = T * points_c
  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Estimates the homography from the given correspondences.
#
# INPUTS:
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   H         [3x3] estimated homography
#
#---------------------------------------------------------
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
  H = load("H.jld2","H")



  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Computes distances for keypoints after transformation
# with the given homography.
#
# INPUTS:
#   H          [3x3] homography
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   d2         distance measure using the given homography
#
#---------------------------------------------------------
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
  points1_h = Common.cart2hom(points1')   # N x 2 -> 3 x N
  points2_h = Common.cart2hom(points2')
  term1 = Common.hom2cart(H*points1_h - points2_h)  #  3 x N -> 2 x N
  term2 = Common.hom2cart(points1_h - (H^-1)*points2_h)
  d2 = [norm(term1[:,i])^2 + norm(term2[:,i])^2 for i in 1:size(term1,2)]
  d2 = Array{Float64,2}(d2')  # N x 1 -> 1 x N
  #d2 = []
  #for i in 1:size(term1,2)
  #  d2 = [d2 norm(term1[:,i])^2+norm(term2[:,i])^2]
  #end
  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end


#---------------------------------------------------------
# Compute the inliers for a given distances and threshold.
#
# INPUTS:
#   distance   homography distances
#   thresh     threshold to decide whether a distance is an inlier
#
# OUTPUTS:
#  n          number of inliers
#  indices    indices (in distance) of inliers
#
#---------------------------------------------------------
function findinliers(distance::Array{Float64,2},thresh::Float64)
  #println(minimum(distance))
  indices = findall(distance[1,:] .< thresh)
  #indices = Array{Int,1}
  n = length(indices)
  return n::Int,indices::Array{Int,1}
end


#---------------------------------------------------------
# RANSAC algorithm.
#
# INPUTS:
#   pairs     potential matches between interest points.
#   thresh    threshold to decide whether a homography distance is an inlier
#   n         maximum number of RANSAC iterations
#
# OUTPUTS:
#   bestinliers   [n x 1 ] indices of best inliers observed during RANSAC
#
#   bestpairs     [4x4] set of best pairs observed during RANSAC
#                 i.e. 4 x [x1 y1 x2 y2]
#
#   bestH         [3x3] best homography observed during RANSAC
#
#---------------------------------------------------------
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)
  k = 4
  bestinliers = Array{Int,1}
  bestpairs = Array{Int,2}
  bestH = Array{Float64,2}
  for i in 1:n
    sample1, sample2 = picksamples(pairs[:,1:2],pairs[:,3:4],k)
    Hi = computehomography(sample1,sample2)
    d2i = computehomographydistance(H,pairs[:,1:2],pairs[:,3:4])
    ni, indices = findinliers(d2i,thresh)

    if i == 1 || ni > length(bestinliers)
      bestinliers = indices
      bestpairs = [sample1 sample2]
      bestH = Hi
    else
      continue
    end
  end
  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end


#---------------------------------------------------------
# Recompute the homography based on all inliers
#
# INPUTS:
#   pairs     pairs of keypoints
#   inliers   inlier indices.
#
# OUTPUTS:
#   H         refitted homography using the inliers
#
#---------------------------------------------------------
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})
  H = computehomography(pairs[inliers,1:2],pairs[inliers,3:4])
  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Show panorama stitch of both images using the given homography.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   H       [3x3] estimated homography between im1 and im2
#
#---------------------------------------------------------
function showstitch(im1::Array{Float64,2},im2::Array{Float64,2},H::Array{Float64,2})
  width = 700
  left = 300
  im2t = zeros(size(im2))
  for r in 1:size(im2t,1)
    for c in left+1:width
      x,y = Common.hom2cart(H*Common.cart2hom([c;r]))
      # Source: https://github.com/JuliaImages/Images.jl/blob/master/src/algorithms.jl
      im2t[r,c-left] = bilinear_interpolation(im2,y,x)
    end
  end
  im = hcat(im1[:,1:left],im2t)
  figure()
  imshow(im,"gray")
  title("Panorama stitching")
  return nothing::Nothing
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  close("all")
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 80000.0 # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("a3p2a.png")
  im2 = PyPlot.imread("a3p2b.png")

  # Convert to double precision
  im1 = Float64.(im1)
  im2 = Float64.(im2)

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("keypoints.jld2")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute chi-square distance  matrix
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)
  @printf(" # of bestinliers : %d", length(bestinliers))

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return nothing::Nothing
end
