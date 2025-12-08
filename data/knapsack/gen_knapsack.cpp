#include <bits/stdc++.h>

using namespace std;

#define MAX_N (ll) (1e2 + 7)
#define MAX_VAL (ll) (1e5 + 7)
#define ll long long

ll n , w;
ll f[MAX_N][MAX_VAL];
ll W[MAX_N] , V[MAX_N];

int main() {
	ios_base::sync_with_stdio(0); cin.tie(NULL);
	cout.tie(NULL);
	cin >> n >> w;
	for (ll i = 1 ; i <= n ; i++) {
		cin >> W[i] >> V[i];
	}
	for (ll i = 1 ; i <= n ; i++) {
		for (ll j = 0 ; j <= w ; j++) {
			if (j < W[i]) f[i][j] = f[i - 1][j];
			else {
				f[i][j] = max(f[i][j] , max(f[i - 1][j] , f[i - 1][j - W[i]] + V[i]));
			}
		}
	}
	cout << f[n][w];
	return 0;
}