-- Bad solution
fib :: Integer -> Integer
fib x
    | x == 1 = 1
    | x == 2 = 2
    | otherwise = fib (x - 1) + fib (x - 2)

sumEvenFib :: Integer -> Integer
sumEvenFib limit = 
    sum (takeWhile (<=limit) (filter even (map fib [1..])))

-- Actual solution
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
