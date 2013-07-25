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


program euler
    implicit none

    print *, "Problem 1: ", problem_1()
    print *, "Problem 2: ", problem_2()


contains

    !! Find the sum of all the multiple of 3 or 5 below 1000.
    integer function problem_1()
        implicit none
        integer :: n, sum
        sum = 0
        do n = 1, 999
            if ((mod(n, 3) == 0) .or. (mod(n, 5) == 0)) then
                sum = sum + n
            endif
        enddo

        problem_1 = sum
    end function problem_1

    !! Find the sum of all even fibonnacci numbers below 4000000.
    integer function problem_2()
        implicit none
        integer :: limit, i1, i2, temp, sum
        limit = 4000000
        i1 = 1
        i2 = 1
        temp = 0
        sum = 0

        do while (i2 < limit)
            temp = i1 + i2
            i1 = i2
            i2 = temp
            if (mod(i2, 2) == 0) then
                sum = sum + i2
            endif
        enddo

        problem_2 = sum
    end function problem_2

end program euler
