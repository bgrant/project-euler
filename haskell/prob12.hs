triangleNums :: [Int]
triangleNums = scanl1 (+) [1..]

divisors :: Int -> [Int]
divisors n = [x | x <- [1..n], n `mod` x == 0]

triangleDivisors = [(n, divisors n) | n <- triangleNums]

prob12 = take 1 (dropWhile (\x -> length (snd x) <= 500) triangleDivisors)
