using Images
using PyPlot
using Statistics
using LinearAlgebra
using Printf

@inline isBigger(a, b) = (a > b)
@inline isSmaller(a, b) = (a < b)

include("Common.jl")

#---------------------------------------------------------
# Load ground truth disparity map
#
# Input:
#   filename:       the name of the disparity file
#
# Outputs:
#   disparity:      [h x w] ground truth disparity
#   mask:           [h x w] validity mask
#---------------------------------------------------------
function loadGTdisaprity(filename)
  disparity_gt = Float64.(PyPlot.imread(filename)) * 256
  mask = disparity_gt.!= 0
  @assert size(mask) == size(disparity_gt)
  return disparity_gt::Array{Float64,2}, mask::BitArray{2}
end

#---------------------------------------------------------
# Calculate NC between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   nc_cost : Normalized Correlation cost
#
#---------------------------------------------------------
function computeNC(patch1, patch2)
  w1 = reshape(patch1,size(patch1,1)*size(patch1,2),1)
  w2 = reshape(patch2,size(patch2,1)*size(patch2,2),1)
  term1 = (w1.-mean(w1))' * (w2.-mean(w2))
  term2 = norm(w1.-mean(w1)) * norm(w2.-mean(w2))
  nc_cost = term1[1] / term2
  return nc_cost::Float64
end

#---------------------------------------------------------
# Calculate SSD between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   ssd_cost : SSD cost
#
#---------------------------------------------------------
function computeSSD(patch1, patch2)
  w1 = reshape(patch1,size(patch1,1)*size(patch1,2),1)
  w2 = reshape(patch2,size(patch2,1)*size(patch2,2),1)
  ssd_cost = sum((w1 - w2).^2)
  return ssd_cost::Float64
end


#---------------------------------------------------------
# Calculate the error of estimated disparity
#
# Inputs:
#   disparity : estimated disparity result, [h x w]
#   disparity_gt : ground truth disparity, [h x w]
#   valid_mask : validity mask, [h x w]
#
# Output:
#   error_disparity : calculated disparity error
#   error_map:  error map, [h x w]
#
#---------------------------------------------------------
function calculateError(disparity, disparity_gt, valid_mask)
  error_map = norm.(disparity-disparity_gt) .* valid_mask
  error_disparity = norm(error_map) / sum(valid_mask)
  @assert size(disparity) == size(error_map)
  return error_disparity::Float64, error_map::Array{Float64,2}
end


#---------------------------------------------------------
# Compute disparity
#
# Inputs:
#   gray_l : a gray version of the left image, [h x w]
#   gray_R : a gray version of the right image, [h x w]
#   max_disp: Maximum disparity for the search range
#   w_size: window size
#   cost_ftn: a cost function for caluclaing the cost between two patches.
#             It can be either computeSSD or computeNC.
#
# Output:
#   disparity : disparity map, [h x w]
#
#---------------------------------------------------------
function computeDisparity(gray_l, gray_r, max_disp, w_size, cost_ftn::Function)
  h,w = size(gray_l)
  disparity = Int.(zeros(h,w))
  r1,r2 = floor.(Int,w_size/2)
  for i in 1+r1 : h-r1
    for j in 1+r2 : w-r2
      patch1 = gray_l[i-r1:i+r1,j-r2:j+r2]
      best_cost = cost_ftn==computeSSD ? Inf : -Inf
      best_d = max_disp
      for k in max(1+r2,j-max_disp) : j
        patch2 = gray_r[i-r1:i+r1,k-r2:k+r2]
        cost = cost_ftn(patch1, patch2)
        if cost_ftn==computeSSD && cost<best_cost
          best_cost = cost
          best_d = j-k
        elseif cost_ftn==computeNC && cost>best_cost
          best_cost = cost
          best_d = j-k
        end
      end
      disparity[i,j] = best_d
    end
  end
  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
#   An efficient implementation
#---------------------------------------------------------
function computeDisparityEff(gray_l, gray_r, max_disp, w_size)
  h,w = size(gray_l)
  disparity = Int.(zeros(h,w))
  r1,r2 = floor.(Int,w_size/2)
  # assign a matrix to store all costs for one single row
  # each column stores costs of sliding windows in range [0,max_disp]
  allcosts = ones(max_disp+1,w)*Inf
  for i in 1+r1 : h-r1
    if i == 1+r1
      # initialize cost matrix for the first row
      for j in 1+r2 : w-r2
        patch1 = gray_l[i-r1:i+r1,j-r2:j+r2]
        #best_cost = Inf
        #best_d = max_disp
        for k in max(1+r2,j-max_disp) : 1 : j # or a larger step
          patch2 = gray_r[i-r1:i+r1,k-r2:k+r2]
          cost = computeSSD(patch1, patch2)
          allcosts[j-k+1,j] = cost
          #if cost<best_cost
          #  best_cost = cost
          #  best_d = j-k
          #end
        end
        #disparity[i,j] = best_d
        best_cost,best_d = findmin(allcosts[:,j])
        disparity[i,j] = best_d-1
      end
    else
      # update cost matrix for the next row
      for j in 1+r2 : w-r2
        patch1_cut = gray_l[i-1-r1,j-r2:j+r2]
        patch1_add = gray_l[i+r1,j-r2:j+r2]
        #best_cost = Inf
        #best_d = max_disp
        for k in max(1+r2,j-max_disp) : 1 : j   # or a larger step
          patch2_cut = gray_r[i-1-r1,k-r2:k+r2]
          patch2_add = gray_r[i+r1,k-r2:k+r2]
          allcosts[j-k+1,j] += - computeSSD(patch1_cut,patch2_cut) + computeSSD(patch1_add,patch2_add)
          # => so that only 2*w_size[2] operations are added instead of w_size[1]*w_size[2] operations
          #cost = allcosts[j-k+1,j]
          #if cost<best_cost
          #  best_cost = cost
          #  best_d = j-k
          #end
        end
        best_cost,best_d = findmin(allcosts[:,j])
        disparity[i,j] = best_d-1
        #disparity[i,j] = best_d
      end
    end
  end
  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
# Problem 2: Stereo matching
#---------------------------------------------------------
function problem2()

  # Define parameters
  w_size = [5 5]
  max_disp = 100
  gt_file_name = "a4p2_gt.png"

  # Load both images
  gray_l, rgb_l = Common.loadimage("a4p2_left.png")
  gray_r, rgb_r = Common.loadimage("a4p2_right.png")

  # Load ground truth disparity
  disparity_gt, valid_mask = loadGTdisaprity(gt_file_name)

  # estimate disparity
  @time disparity_ssd = computeDisparity(gray_l, gray_r, max_disp, w_size, computeSSD)
  @time disparity_nc = computeDisparity(gray_l, gray_r, max_disp, w_size, computeNC)


  # Calculate Error
  error_disparity_ssd, error_map_ssd = calculateError(disparity_ssd, disparity_gt, valid_mask)
  @printf(" disparity_SSD error = %f \n", error_disparity_ssd)
  error_disparity_nc, error_map_nc = calculateError(disparity_nc, disparity_gt, valid_mask)
  @printf(" disparity_NC error = %f \n", error_disparity_nc)

  figure()
  subplot(2,1,1), imshow(disparity_ssd, interpolation="none"), axis("off"), title("disparity_ssd")
  subplot(2,1,2), imshow(error_map_ssd, interpolation="none"), axis("off"), title("error_map_ssd")
  gcf()

  figure()
  subplot(2,1,1), imshow(disparity_nc, interpolation="none"), axis("off"), title("disparity_nc")
  subplot(2,1,2), imshow(error_map_nc, interpolation="none"), axis("off"), title("error_map_nc")
  gcf()

  figure()
  imshow(disparity_gt)
  axis("off")
  title("disparity_gt")
  gcf()

  @time disparity_ssd_eff = computeDisparityEff(gray_l, gray_r, max_disp, w_size)
  error_disparity_ssd_eff, error_map_ssd_eff = calculateError(disparity_ssd_eff, disparity_gt, valid_mask)
  @printf(" disparity_SSD_eff error = %f \n", error_disparity_ssd_eff)

  figure()
  subplot(2,1,1), imshow(disparity_ssd_eff, interpolation="none"), axis("off"), title("disparity_ssd_eff")
  subplot(2,1,2), imshow(error_map_ssd_eff, interpolation="none"), axis("off"), title("error_map_ssd_eff")
  gcf()

  return nothing::Nothing
end
