-- `Project Euler <http://www.projecteuler.net>`_ solutions in Haskell.
--
-- :author: Robert David Grant <robert.david.grant@gmail.edu>
--
-- :copyright:
--   Copyright 2011 Robert David Grant
--
--   Licensed under the Apache License, Version 2.0 (the "License"); you
--   may not use this file except in compliance with the License.  You
--   may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--   Unless required by applicable law or agreed to in writing, software
--   distributed under the License is distributed on an "AS IS" BASIS,
--   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
--   implied.  See the License for the specific language governing
--   permissions and limitations under the License.


-- Functions --
rFib :: Integer -> Integer
rFib x
    | x == 1 = 1
    | x == 2 = 2
    | otherwise = rFib (x - 1) + rFib (x - 2)

sumEvenFib :: Integer -> Integer
sumEvenFib limit = 
    sum (takeWhile (<=limit) (filter even (map rFib [1..])))

nextFib :: [Integer] -> [Integer]
nextFib [] = [1]
nextFib [1] = [2,1]
nextFib (x2:x1:xs) = (x1 + x2):x2:x1:xs

allFibs :: Integer -> [Integer] -> [Integer]
allFibs limit xs = 
    if (head xs) <= limit
    then allFibs limit (nextFib xs)
    else (tail xs)

sumEvens :: [Integer] -> Integer
sumEvens xs = sum $ filter even xs

-- Problems --
-- problem2 :: Integer -> Integer
-- problem2 limit = sumEvens (allFibs limit)
