using Images
using PyPlot
using Test
using LinearAlgebra
using FileIO

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
  points_hom = vcat(points,ones(1,size(points,2)))
  return points_hom::Array{Float64,2}
end


# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
  points_cart = points[1:end-1,:] ./ points[end,:]'
  return points_cart::Array{Float64,2}
end


# Translation by v
function gettranslation(v::Array{Float64,1})
  T = Matrix{Float64}(I,4,4)
  T[1:3,end] = v
  return T::Array{Float64,2}
end

# Source: https://en.wikipedia.org/wiki/Rotation_matrix
# Rotation of d degrees around x axis
function getxrotation(d::Int)
  Rx = Matrix{Float64}(I,4,4)
  r = deg2rad(d)
  Rx[2:3,2:3] = [cos(r) -sin(r); sin(r) cos(r)]
  return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)
  Ry = Matrix{Float64}(I,4,4)
  r = deg2rad(d)
  Ry[1:2:3,1:2:3] = [cos(r) sin(r); -sin(r) cos(r)]
  return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)
  Rz = Matrix{Float64}(I,4,4)
  r = deg2rad(d)
  Rz[1:2,1:2] = [cos(r) -sin(r); sin(r) cos(r)]
  return Rz::Array{Float64,2}
end


# Central projection matrix (including camera intrinsics)
function getcentralprojection(principal::Array{Int,1}, focal::Int)
  K = [focal 0 principal[1] 0;
      0 focal principal[2] 0;
      0 0 1 0.0]                # 0.0 to widen type of matrix to Float64
  return K::Array{Float64,2}
end


# Return full projection matrix P and full model transformation matrix M
function getfullprojection(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
  M = Rz*Rx*Ry*T  # M = [R T; 0 1] (transformation matrix 4x4)
  P = V*M         # V = K (projection matrix 3x4)
  return P::Array{Float64,2},M::Array{Float64,2}
end



# Load 2D points
function loadpoints()
  points = load("obj2d.jld2","x")
  return points::Array{Float64,2}
end


# Load z-coordinates
function loadz()
  z = load("zs.jld2","Z")
  return z::Array{Float64,2}
end


# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
  P2d_hom = cart2hom(P2d) .* z  # [x' y'] => [x' y' 1] => [x y z]
  K = P[:,1:3]                  # P2d_hom = P * P3d_hom = K * [I 0] * P3d_hom = K * P3d
  P3d = K \ P2d_hom             # => P3d = (K^-1) * P2d_hom
  return P3d::Array{Float64,2}
end


# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
  P3d_hom = cart2hom(P3d)
  X = A \ P3d_hom           # homogeneous coordinates
  #X = hom2cart(X)
  return X::Array{Float64,2}
end


# Plot 2D points
function displaypoints2d(points::Array{Float64,2})
  figure()
  scatter(points[1,:],points[2,:])  # or plot()
  xlabel("Image X")
  ylabel("Image Y")
  return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})
  figure()
  scatter3D(points[1,:],points[2,:],points[3,:])
  xlabel("World X")
  ylabel("World Y")
  zlabel("World Z")
  return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(P::Array{Float64,2}, X::Array{Float64,2})
  X_hom = cart2hom(X)
  P2d_hom = P*X_hom
  P2d = hom2cart(P2d_hom)
  return P2d::Array{Float64,2}
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

#= It is necessary to provide the z-coordinates of all points,
in order to reconstruct the depth of the 3D information.
Without them, a single 2D point can only be recovered into a ray,
which contains an infinite number of 3D points.
=#

#= Changing the order of the transformations is not commutative,
because the multiplication of matrices is not commutative.
=#
