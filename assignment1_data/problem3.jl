using Images
using PyPlot
using Test
using LinearAlgebra
using FileIO

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})


  return points_hom::Array{Float64,2}
end


# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})

  return points_cart::Array{Float64,2}
end


# Translation by v
function gettranslation(v::Array{Float64,1})

  return T::Array{Float64,2}
end

# Rotation of d degrees around x axis
function getxrotation(d::Int)

  return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)

  return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)

  return Rz::Array{Float64,2}
end


# Central projection matrix (including camera intrinsics)
function getcentralprojection(principal::Array{Int,1}, focal::Int)

  return K::Array{Float64,2}
end


# Return full projection matrix P and full model transformation matrix M
function getfullprojection(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})

  return P::Array{Float64,2},M::Array{Float64,2}
end



# Load 2D points
function loadpoints()

  return points::Array{Float64,2}
end


# Load z-coordinates
function loadz()

  return z::Array{Float64,2}
end


# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})

  return P3d::Array{Float64,2}
end


# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})

  return X::Array{Float64,2}
end


# Plot 2D points
function displaypoints2d(points::Array{Float64,2})

  return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})

  return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(P::Array{Float64,2}, X::Array{Float64,2})

  return P2d:::Array{Float64,2}
end



#= Problem 2
Projective Transformation =#

function problem3()
  # parameters
  t               = [6.7; -10; 4.2]
  principal_point = [9; -7]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(-45)
  Rx = getxrotation(120)
  Rz = getzrotation(-10)

  # central projection including camera intrinsics
  K = getcentralprojection(principal_point,focal_length)

  # full projection and model matrix
  P,M = getfullprojection(T,Rx,Ry,Rz,K)

  # load data and plot it
  points = loadpoints()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  Xt = invertprojection(K,points,z)
  Xh = inverttransformation(M,Xt)

  worldpoints = hom2cart(Xh)
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(P,worldpoints)
  displaypoints2d(points2)

  @test points ≈ points2
  return
end
