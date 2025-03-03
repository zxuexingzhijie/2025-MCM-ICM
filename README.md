
### 2025 MCM Problem C: 
[中文版]: README_zh.md
### "Olympic Medal Prediction Model" Code and Partial Thought Process

#### For the Olympic medal prediction model, our ideas are as follows:

- ***Main Model: XGBoost Model***
   We utilize this gradient boosting tree algorithm to iteratively train a series of weak learners (decision trees) and combine them into a strong learner. This approach handles static features and simple temporal features. By tuning hyperparameters and evaluating the model, we use it as our main model to predict the number of medals and the probability of winning a medal for the first time.

![image-20250302132328851](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031418321.png)

- ***Submodel 1: SVR Model***
   We established a smooth regression function by minimizing the objective function. Combined with our main model, this approach was used to predict medals for countries with limited historical data and moderate feature dimensions (such as the Netherlands and Finland). Additionally, we analyzed the nonlinear impact of changes in the number of sports events on medal distribution.

![image-20250302132340935](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031419397.png)

- ***Submodel 2: Random Forest Model***
   We used the bootstrap method and random feature selection to generate multiple decision trees and averaged their results. By analyzing the importance of event-related features within the random forest model, we explored the relationship between different events and the number of medals won by various countries.

![image-20250302132540645](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031419715.png)

#### Prediction Model Workflow

![image-20250302132051836](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031419550.png)

#### Data preprocessing component

![image-20250302132723521](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031419727.png)

![image-20250302132731319](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031420320.png)

![image-20250302132737244](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031420379.png)

![image-20250302132745675](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031420916.png)

### Model Results:

- #### Medal Count Prediction

By constructing analytical feature indicators and preprocessing historical data, our trained XG-S-R model generated medal count predictions for various countries and regions in future Olympic Games. Here, we present the countries that rank among the top in medal counts. Similarly...

![image-20250302132814818](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031420685.png)

- #### Analysis of Changes in Medal Performance

From the chart, we can see that countries such as the United States, China, and the United Kingdom are most likely to make progress, with China and the United States showing significantly greater improvements compared to other countries. On the other hand, countries or independent athletes with weaker performances, such as some independent athletes, Hungary, and Australia, are expected to perform worse than in 2024.

![image-20250302132931536](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031421278.png)

![image-20250302132937933](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031421867.png)

### Prediction of New Medal-Winning Countries

The chart presents the top 10 countries most likely to win their first gold medal. The top three—Afghanistan, North Macedonia, and the Republic of Niger—each have a probability exceeding 60%. The other countries also have a probability of over 50% of winning their first gold medal.

![image-20250302133127449](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031421491.png)

#### Analysis of the Impact of Olympic Event Settings

We solved the correlation matrix and visualized it using a correlation heatmap.

1. **Relationship Between the Total Number of Events and Medal Counts**: The correlation coefficients between the total number of events and gold, silver, bronze, and total medal counts are all close to zero. This indicates that changes in the number of Olympic events have almost no direct impact on the number of medals won by different countries.
2. **High Correlation Among Medal Types**: There is a strong positive correlation (correlation coefficient: 0.88–0.92) among gold, silver, and bronze medals. This suggests that a country’s strength in one type of medal is usually accompanied by strength in others, reflecting its overall competitive capability.
3. **Key Drivers of Total Medal Count**: The total number of medals is highly correlated with gold, silver, and bronze counts (correlation coefficient: 0.96–0.97). Gold and silver medals contribute the most to the total medal count, indicating that a country's overall performance is determined by its results across different medal types.

In summary, the distribution of Olympic medals is primarily related to a country’s overall competitive strength rather than the total number of events. Additionally, there is a strong synergy between different medal types and the total medal count, reinforcing the "strong-get-stronger" effect.

![image-20250302133203266](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031422657.png)

#### Exploration of the host country effect

![image-20250302133221650](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031422246.png)

![image-20250302133226148](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031422829.png)

From the analysis of these two charts, we can conclude the following:

- **The host country has a significant medal advantage in the hosting year**: As shown in Figure 15, the total number of medals won by the host country is significantly higher in hosting years compared to non-hosting years. This suggests that host countries may benefit from home advantage, leading to more medal wins.
- **The gap between host and non-host countries may widen**: In Figure 14, the medal count curve for host countries shows distinct peaks in hosting years (e.g., 1900, 1920), whereas the curve for non-host countries remains relatively stable. This indicates that the advantage of host countries might become more pronounced over time.

Through this visualization analysis, we can conclude that being the host country does indeed impact the number of medals won. Hosting the Olympics increases the probability of athletes winning medals and enhances the overall medal count for the country.



#### An Exploration of the Great Coach Effect and Its Olympic Suggestions

![image-20250302133842575](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031423934.png)

![image-20250302133849498](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031423000.png)

![image-20250302133857605](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031423802.png)

![image-20250302133903058](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031423212.png)

![image-20250302133912623](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031423573.png)

![image-20250302133919643](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031424744.png)

We selected China, the United States, and Germany for analysis in the Summer Olympic Games, visualizing their athlete data, including participation in different sports, medal achievements, and medal efficiency.

### (1) Data Comparison and Analysis

- **Dominant Sports by Country**: China excels in table tennis, diving, and gymnastics, with table tennis achieving a remarkable medal efficiency of 2.26 medals per athlete. Denmark has a strong advantage in athletics and swimming, with a high number of medals and athlete participation. Germany shows competitive strength in rowing, field hockey, and equestrian events, with rowing exhibiting particularly high medal efficiency.
- **Athlete Participation and Medal Achievements**: Denmark has a higher number of participating athletes across multiple sports and secures a significant number of medals. China and Germany have concentrated advantages in specific sports, with relatively lower participation and medal counts in certain events.
- **Differences in Medal Efficiency**: Medal efficiency varies significantly across different sports for each country. China achieves high efficiency in diving and table tennis, while some ball sports have lower efficiency. Denmark's swimming efficiency reaches 1.576471, but some niche sports show zero efficiency. Germany's canoeing and equestrian events have high efficiency, whereas some team sports perform less efficiently. These differences reflect varying levels of development and resource allocation among countries.

### (2) "Great Coach" Effect and Recommendations

- **Effect Demonstration**: Based on the model, hiring elite coaches leads to an average increase of 1.1 medals per Olympic cycle for technical individual sports and 1.8 medals for team sports. This effect is evident in all three countries analyzed. For instance, China's gymnastics, the U.S. shooting, and Germany’s canoeing have shown medal increases with coaching improvements. In team sports, China’s women's volleyball, Denmark’s women's volleyball, and Germany’s men's handball have also benefited from the "great coach" effect.
- **Investment Priority Considerations**: Among the three countries, team sports tend to receive higher investment priority due to their larger potential medal increase (1.8 medals per Olympic cycle). Sports like women's volleyball and men's handball are prime examples. In technical individual events, disciplines with strong potential but not yet at the highest efficiency—such as China’s gymnastics and Germany’s canoeing—also hold high investment value. While endurance-based sports have a lower effect coefficient, their large athlete base provides room for improvement, though they generally rank lower in investment priority.

#### Comparison with the Ridge Model:

- ##### Ridge Model Fit:

![image-20250302134330727](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031425726.png)

##### ![image-20250302134306639](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031425085.png) XG-S-R Model Fit:

![image-20250302134415547](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031426495.png)

![image-20250302134423843](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031426935.png)

- The Ridge model performs well on certain samples but exhibits systematic bias, underestimating high medal counts and overestimating low medal counts.
- The XG-S-R model demonstrates overall strong predictive performance on the data.

| Model Name   | MAE (Gold) | R² (Gold) | MAE (Total) | R² (Total) |
| ------------ | ---------- | --------- | ----------- | ---------- |
| XG-S-R Model | 1.03       | 0.86      | 3.91        | 0.81       |
| Ridge Model  | 49.29      | 0.49      | 333.88      | 0.48       |

### Model Evaluation

![image-20250302134618920](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503031426999.png)

The XG-S-R model remains stable in low-noise environments (standard deviation <0.2), with controlled error increases. In high-noise environments, maintaining stability is achievable by enhancing the data-cleaning process.

