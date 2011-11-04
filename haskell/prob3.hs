-- primeFactors :: Integer -> [Integers]
-- primeFactors n
--     | n < 2     = []
--     | otherwise = maximum $ iterPrimes n 2 []
-- 
-- iterPrimes :: Integer -> Integer -> [Integers] -> [Integers]
-- iterPrimes n i knownPrimes
--     | i > (sqrt n) = knownPrimes
--     | n `div` i == 0 
--     | otherwise    = iterPrimes n (i+1) knownPrimes

-- primeFactors :: Integer -> [Integers]
-- primeFactors n = [x <- [2..sqrt(n)], 

-- upperBound :: Integer -> Integer
-- upperBound n = ceiling . (/2) $ fromInteger n
-- 
-- isPrime :: Integer -> Bool
-- isPrime 2 = True
-- isPrime n = null $ filter (\x -> n `mod` x == 0) [2 .. upperBound n]
-- 
-- allPrimeFactors :: Integer -> [Integer]
-- allPrimeFactors n
--     | isPrime n = [n]
--     | otherwise  = filter (\x -> and [isPrime x, n `mod` x == 0]) [2 .. upperBound n]

primeFactors :: Integer -> [Integer]
primeFactors n  = iterFactors n 2 []
    where iterFactors n ctr factors
            | isPrime n         = n:factors
            | n `mod` ctr == 0  = iterFactors (n `div` ctr) ctr (ctr:factors)
            | otherwise         = iterFactors n (ctr+1) factors
