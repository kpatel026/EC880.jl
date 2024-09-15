module primitives

    ! defines primitives of the model
    implicit none 

    ! preference parameters
    real(kind=8), parameter :: beta = 0.9932
    real(kind=8), parameter :: alpha = 1.50

    ! grids used in model 
    integer, parameter :: ns = 2
    integer, parameter :: na = 501 

    ! declare grids
    real(kind = 8) :: s_grid(ns) = [1.0, 0.5] ! employment state grid
    real(kind = 8) :: pi(ns, ns) = reshape([0.97, 0.03, 0.5, 0.5], shape(pi)) ! employment state transition probabilities
    real(kind = 8) :: a_grid(na) 
end module primitives

module results
    use primitives
    ! declare grids for equilibrium objects 

    real(kind = 8) :: val_func(na, ns)
    real(kind = 8) :: pol_func(na, ns)
    real(kind = 8) :: mu(na, ns)
    real(kind = 8) :: q
end module results 



program test

  use stdlib_math
  use primitives
  use results
  implicit none

  
  call linspace(-2.0, 5.0, na, a_grid)

end program test