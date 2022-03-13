#include <gtest/gtest.h>

TEST(MeaningOfLifeTest, TheAnswerToLifeTheUniverseAndEverything) {
	constexpr auto kMeaningOfLife = 42;
	ASSERT_EQ(kMeaningOfLife, 42);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
