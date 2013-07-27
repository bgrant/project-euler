c Project Euler <http://www.projecteuler.net> solutions in Fortran 77.
c
c :author: Robert David Grant <robert.david.grant@gmail.com>
c
c :copyright:
c   Copyright 2013 Robert David Grant
c
c   Licensed under the Apache License, Version 2.0 (the "License"); you
c   may not use this file except in compliance with the License.  You
c   may obtain a copy of the License at
c
c      http://www.apache.org/licenses/LICENSE-2.0
c
c   Unless required by applicable law or agreed to in writing, software
c   distributed under the License is distributed on an "AS IS" BASIS,
c   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
c   implied.  See the License for the specific language governing
c   permissions and limitations under the License.


c Project Euler problems in Fortran 77.
      program euler
      integer problem_1
      integer problem_2

      print *, "Problem 1: ", problem_1(999)
      print *, "Problem 2: ", problem_2(4000000)

      stop
      end


c Find the sum of all the multiples of 3 or 5 below n.
      integer function problem_1(n)
      integer n, i, total
      total = 0
      do 10 i=1, n
          if ((mod(i, 3) == 0) .or. (mod(i, 5) == 0)) then
              total = total + i
          endif
10    continue
      problem_1 = total
      return
      end


c Find the sum of all even Fibonnacci numbers below n.
      integer function problem_2(n)
      integer total, n, i1, i2, temp

      i1 = 0
      i2 = 1
      total = 0
10    if (i2 < n) then
          temp = i2
          i2 = i1 + i2
          i1 = temp
          if (mod(i2, 2) == 0) then
              total = total + i2
          endif
          goto 10
      endif
      problem_2 = total
      return
      end
