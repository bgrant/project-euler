!! Project Euler <http://www.projecteuler.net> solutions in Fortran.
!!
!! :author: Robert David Grant <robert.david.grant@gmail.com>
!!
!! :copyright:
!!   Copyright 2013 Robert David Grant
!!
!!   Licensed under the Apache License, Version 2.0 (the "License"); you
!!   may not use this file except in compliance with the License.  You
!!   may obtain a copy of the License at
!!
!!      http://www.apache.org/licenses/LICENSE-2.0
!!
!!   Unless required by applicable law or agreed to in writing, software
!!   distributed under the License is distributed on an "AS IS" BASIS,
!!   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
!!   implied.  See the License for the specific language governing
!!   permissions and limitations under the License.


!! Find the sum of all the multiple of 3 or 5 below 1000.
program problem_1
implicit none

integer :: n, sum

n = 0
sum = 0

do while (n .lt. 999)
    n = n + 1
    if ((mod(n, 3) .eq. 0) .or. (mod(n, 5) .eq. 0)) then
        sum = sum + n
    endif
enddo

print *, "Problem 1:", sum

end program problem_1
