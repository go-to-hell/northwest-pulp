\* Transportation_Problem *\
Minimize
Sum_of_Transporting_Costs: 10 Route_A_W + 10 Route_A_X + 40 Route_A_Y
 + 10 Route_A_Z + 40 Route_B_W + 10 Route_B_X + 30 Route_B_Y + 20 Route_B_Z
 + 60 Route_C_W + 30 Route_C_X + 20 Route_C_Y + 10 Route_C_Z
Subject To
Sum_of_Products_into_Store_W: Route_A_W + Route_B_W + Route_C_W >= 10
Sum_of_Products_into_Store_X: Route_A_X + Route_B_X + Route_C_X >= 40
Sum_of_Products_into_Store_Y: Route_A_Y + Route_B_Y + Route_C_Y >= 20
Sum_of_Products_into_Store_Z: Route_A_Z + Route_B_Z + Route_C_Z >= 40
Sum_of_Products_out_of_Warehouse_A: Route_A_W + Route_A_X + Route_A_Y
 + Route_A_Z <= 30
Sum_of_Products_out_of_Warehouse_B: Route_B_W + Route_B_X + Route_B_Y
 + Route_B_Z <= 50
Sum_of_Products_out_of_Warehouse_C: Route_C_W + Route_C_X + Route_C_Y
 + Route_C_Z <= 30
Bounds
 0 <= Route_A_W
 0 <= Route_A_X
 0 <= Route_A_Y
 0 <= Route_A_Z
 0 <= Route_B_W
 0 <= Route_B_X
 0 <= Route_B_Y
 0 <= Route_B_Z
 0 <= Route_C_W
 0 <= Route_C_X
 0 <= Route_C_Y
 0 <= Route_C_Z
Generals
Route_A_W
Route_A_X
Route_A_Y
Route_A_Z
Route_B_W
Route_B_X
Route_B_Y
Route_B_Z
Route_C_W
Route_C_X
Route_C_Y
Route_C_Z
End
