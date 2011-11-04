isPalindrome n = (show n) == (reverse (show n))

allPalindromes = filter isPalindrome [x * y | x <- [999,998..900], y <- [999,998..900]]
