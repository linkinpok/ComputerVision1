using Images
using LinearAlgebra
using PyPlot
using Printf
using Statistics

# Load images from the yale_faces directory and return a MxN data matrix,
# where M is the number of pixels per face image and N is the number of images.
# Also return the dimensions of a single face image and the number of all face images
function loadfaces()
  ids = 38
  shades = 20
  n = ids * shades
  dim1 = 96
  dim2 = 84
  m = dim1 * dim2
  facedim = [dim1 dim2]
  data = zeros(m,n)
  col = 1
  for id in 1:ids
    for shade in 1:shades
      fullpath = string("yale_faces/yaleBs",lpad(id,2,'0'),"/",lpad(shade,2,'0'),".pgm")
      img = load(fullpath)
      data[:,col] = img[:]
      col += 1
    end
  end
  return data::Array{Float64,2},facedim::Array{Int},n::Int
end

# Apply principal component analysis on the data matrix.
# Return the eigenvectors of covariance matrix of the data, the corresponding eigenvalues,
# the one-dimensional mean data matrix and a cumulated variance vector in increasing order.
function computepca(data::Array{Float64,2})
  # Source: https://docs.julialang.org/en/v1/stdlib/Statistics/index.html
  # find mean vector
  E = zeros(size(data,1))
  mean!(E,data)
  # subtract the mean
  X = data.-E
  # compute eigenvectors and eigenvalues
  eig = svd(X)
  U = eig.U
  S = eig.S   # sorted in descending order
  lambda = S.^2/length(S)
  # calculate mean face
  mu = reshape(E,96,84)
  # calculate cumulative variance vector
  # note: the variance of eigenvectors of a covariance matrix equal to its eigenvalues
  # source: https://de.mathworks.com/matlabcentral/fileexchange/26791-cumulative-mean-and-variance
  cumvar = [sum(lambda[1:i]) for i in 1:length(lambda)] / length(lambda)
  return U::Array{Float64,2},lambda::Array{Float64,1},mu::Array{Float64,2},cumvar::Array{Float64,1}
end

# Plot the cumulative variance of the principal components
function plotcumvar(cumvar::Array{Float64,1})
  figure()
  plot(cumvar)
  title("Cumulative Variance")
  return nothing::Nothing
end


# Compute required number of components to account for (at least) 75/99% of the variance
function computecomponents(cumvar::Array{Float64,1})
  # normalize cumulative variance vector
  cumvar_norm = cumvar / maximum(cumvar)
  n75 = length(cumvar_norm[cumvar_norm.<=0.75]) + 1
  n99 = length(cumvar_norm[cumvar_norm.<=0.99]) + 1
  return n75::Int,n99::Int
end


# Display the mean face and the first 10 Eigenfaces in a single figure
function showeigenfaces(U::Array{Float64,2},mu::Array{Float64,2},facedim::Array{Int})
  figure()
  subplot(4,3,1), imshow(mu,"gray"), title("Mean Face")
  for i in 1:10
    subplot(4,3,i+2)
    imshow(reshape(U[:,i],facedim[1],facedim[2]),"gray")
    title(string("Eigenface ",i))
  end
  return nothing::Nothing
end


# Fetch a single face with given index out of the data matrix
function takeface(data::Array{Float64,2},facedim::Array{Int},n::Int)
  face = reshape(data[:,n],facedim[1],facedim[2])
  return face::Array{Float64,2}
end


# Project a given face into the low-dimensional space with a given number of principal
# components and reconstruct it afterwards
function computereconstruction(faceim::Array{Float64,2},U::Array{Float64,2},mu::Array{Float64,2},n::Int)
  proj = U[:,1:n]'*(faceim[:]-mu[:])
  recon = U[:,1:n]*proj+mu[:]
  recon = reshape(recon,size(faceim,1),size(faceim,2))
  return recon::Array{Float64,2}
end

# Display all reconstructed faces in a single figure
function showreconstructedfaces(faceim, f5, f15, f50, f150)
  figure()
  faces = [f5,f15,f50,f150]
  for i in 1:length(faces)
    subplot(3,2,i)
    imshow(faces[i],"gray")
    title(string("Reconstruction ",i))
  end
  subplot(3,2,5), imshow(faceim,"gray"), title("Original")
  return nothing::Nothing
end

# Problem 2: Eigenfaces

function problem2()
  # load data
  data,facedim,N = loadfaces()

  # compute PCA
  U,lambda,mu,cumvar = computepca(data)

  # plot cumulative variance
  plotcumvar(cumvar)

  # compute necessary components for 75% / 99% variance coverage
  n75,n99 = computecomponents(cumvar)
  println(@sprintf("Necssary components for 75%% variance coverage: %i", n75))
  println(@sprintf("Necssary components for 99%% variance coverage: %i", n99))

  # plot mean face and first 10 Eigenfaces
  showeigenfaces(U,mu,facedim)

  # get a random face
  faceim = takeface(data,facedim,rand(1:N))

  # reconstruct the face with 5, 15, 50, 150 principal components
  f5 = computereconstruction(faceim,U,mu,5)
  f15 = computereconstruction(faceim,U,mu,15)
  f50 = computereconstruction(faceim,U,mu,50)
  f150 = computereconstruction(faceim,U,mu,150)

  # display the reconstructed faces
  showreconstructedfaces(faceim, f5, f15, f50, f150)

  return
end
