sumOfSquares :: [Integer] -> Integer
sumOfSquares xs = sum $ map (^2) xs

squareOfSums :: [Integer] -> Integer
squareOfSums xs = (sum xs) ^ 2
