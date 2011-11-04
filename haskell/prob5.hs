divides :: Integer -> Integer -> Bool
divides x n = n `mod` x == 0

rangeDivides :: [Integer] -> Integer -> Bool
rangeDivides xs n = all (\x -> x) [x `divides` n | x <- xs]

firstDivides :: [Integer] -> [Integer] -> Integer 
firstDivides xs ns = head . snd $ break (\n -> rangeDivides xs n) ns
