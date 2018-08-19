/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: main.cpp 2720 2017-12-30 21:04:35+09:00 nowatari $
 * ======================================================================= */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <windows.h>

#include <vector>
#include <random>

#include "resource.h"

#include "NeuralNetTest/NeuralNet.h"
#include "teacherData.h"

#define APP_NAME TEXT("Othello")

// ボードの大きさ.
#define BOARD_SIZE	(8)

// ボードの状態.
typedef enum BOARD_STATE_
{
	EMPTY	= -1,
	BLACK,
	WHITE,
} BOARD_STATE;

// 手の記録.
typedef struct record_tag
{
	int	x, y;
	int	bw;
} record_t;

static int	board[BOARD_SIZE][BOARD_SIZE];
static int	pieceNum[2];
static int	turn;
static int	manColor, comColor;

static int comCursorX, comCursorY;
static int manCursorX, manCursorY;

static HWND	g_hWnd;
static HPEN	hPen1, hPen2;

static record_t	record[BOARD_SIZE*BOARD_SIZE-4];

static const int gridBase	= 10;
static const int gridSize	= 70;

static bool learn	= false;

NeuralNet	Net;

typedef struct teacherLog_tag
{
	std::vector<double>	input;
	int					id;
} teacherLog_t;
std::vector<teacherLog_t>	bwLog[2];

// 白と黒の入れ替え
#define SwapBW(x)	((x)^1)

// 連番とX・Yの入れ替え.
#define IDtoX(id)		((id)/BOARD_SIZE)
#define IDtoY(id)		((id)%BOARD_SIZE)
#define MakeID(x, y)	((x)*BOARD_SIZE+y)

/*----------------------------------------------------------------------
 * ボードを初期化する関数.
 *----------------------------------------------------------------------*/
static void format(void)
{
	turn = 0;

	/* 盤の全てを空にする */
	for (int x = 0; x < BOARD_SIZE; ++x)
	{
		for (int y = 0; y < BOARD_SIZE; ++y)
			board[x][y] = EMPTY;
	}

	/* 最初の４枚を配置する */
	board[BOARD_SIZE/2-1][BOARD_SIZE/2-1]	= board[BOARD_SIZE/2][BOARD_SIZE/2]
											= BLACK;
	board[BOARD_SIZE/2-1][BOARD_SIZE/2]		= board[BOARD_SIZE/2][BOARD_SIZE/2-1]
											= WHITE;

	pieceNum[BLACK]	= pieceNum[WHITE]
					= 2;

	manCursorX	= manCursorY	= -1;
	comCursorX	= comCursorY	= -1;
}

/*----------------------------------------------------------------------
 * ひっくり返すサブ関数.
 *----------------------------------------------------------------------*/
static int subCheck(int (*pBoard)[BOARD_SIZE][BOARD_SIZE],
					int fX,
					int fY,
					int dX,
					int dY,
					int bw)
{
	int	 disbw, count;

	disbw	= SwapBW(bw);
	count	= 0;

	for (int x = fX, y = fY;
		 (x >= 0 && x < BOARD_SIZE)
	  && (y >= 0 && y < BOARD_SIZE);
		 x += dX, y += dY)
	{
		// 自分と同じ駒が見つかったとき
		if ((*pBoard)[x][y] == bw)
		{
			if (count == 0)
				break;

			if ((abs(x - fX) != count)
			&&  (abs(y - fY) != count))
				break;

			for (int i = fX, j = fY ;
				 i != x || j != y;
				 i += dX, j += dY)
			{
				(*pBoard)[i][j]	= bw;
			}

			return (count);
		}
		// 自分と異なる駒の時
		else if((*pBoard)[x][y] == disbw)
		{
			++count;
		}
		else{
			break;
		}
	}

	return (0);
}

/*----------------------------------------------------------------------
 * おける場所を調べてひっくり返す関数
 *----------------------------------------------------------------------*/
static int check(int (*pBoard)[BOARD_SIZE][BOARD_SIZE],
				 int bw,
				 int x,
				 int y)
{
	int place = 0;

	/* 範囲外においたとき */
	if ((x < 0)
	||  (x > BOARD_SIZE)
	||  (y < 0)
	||  (y > BOARD_SIZE))
		return (0);

	/* 置いた位置が空じゃないとき */
	if ((*pBoard)[x][y] != EMPTY)
		return (0);

	//横に調べる
	place	+= subCheck(pBoard, x+1, y  , +1,  0, bw);	//右に調べる
	place	+= subCheck(pBoard, x-1, y  , -1,  0, bw);	//左に調べる

	//縦に調べる
	place	+= subCheck(pBoard, x  , y+1,  0, +1, bw);	//下に調べる
	place	+= subCheck(pBoard, x  , y-1,  0, -1, bw);	//上に調べる

	//斜めに調べる
	place	+= subCheck(pBoard, x+1, y+1, +1, +1, bw);	//右下に調べる
	place	+= subCheck(pBoard, x-1, y+1, -1, +1, bw);	//左下に調べる
	place	+= subCheck(pBoard, x+1, y-1, +1, -1, bw);	//右上に調べる
	place	+= subCheck(pBoard, x-1, y-1, -1, -1, bw);	//左上に調べる

	return (place);
}

/*----------------------------------------------------------------------
 * おける場所の数を調べる関数.
 *----------------------------------------------------------------------*/
static int preCheck(int bw)
{
	int	x, y, num;
	int	copyBoard[BOARD_SIZE][BOARD_SIZE];

	num 	= 0;
	for (x = 0; x < BOARD_SIZE; ++x)
	{
		for (y = 0; y < BOARD_SIZE; ++y)
		{
			memcpy(copyBoard, board, sizeof(copyBoard));

			if (check(&copyBoard, bw, x, y) > 0)
				++num;
		}
	}

	return (num);
}

/*----------------------------------------------------------------------
 * コンピュータ思考関数.
 *----------------------------------------------------------------------*/
static int Cpu(int bw)
{
	typedef struct putCandidacy_tag
	{
		int		id;
		double	ratio;
	} putCandidacy_t;

	std::vector<putCandidacy_t>			array;
	std::random_device					rd;
	std::mt19937						mt(rd());
	std::uniform_real_distribution<>	dist(0.0, 1.0);
	double								total	= 0.0;
	teacherLog_t						log;

	if (learn)
	{
		log.input.resize(BOARD_SIZE*BOARD_SIZE*2);

		for (int x = 0; x < BOARD_SIZE; ++x)
		{
			for (int y = 0; y < BOARD_SIZE; ++y)
			{
				int	myID	= MakeID(x, y);
				int	emyID	= myID + BOARD_SIZE*BOARD_SIZE;

				if (board[x][y] == EMPTY)
				{
					log.input[myID]		= 0.0;
					log.input[emyID]	= 0.0;
				}else if(board[x][y] == bw)
				{
					log.input[myID]		= 1.0;
					log.input[emyID]	= 0.0;
				}else{
					log.input[myID]		= 0.0;
					log.input[emyID]	= 1.0;
				}
			}
		}
	}

	// ニューラルネット配置.
	if (!learn || (dist(mt) >= 0.2))
	{
		std::vector<double>	input;
		std::vector<double>	output;

		input.resize(BOARD_SIZE*BOARD_SIZE*2);
		output.resize(BOARD_SIZE*BOARD_SIZE);

		for (int x = 0; x < BOARD_SIZE; ++x)
		{
			for (int y = 0; y < BOARD_SIZE; ++y)
			{
				int	myID	= MakeID(x, y);
				int	emyID	= myID + BOARD_SIZE*BOARD_SIZE;

				if (board[x][y] == EMPTY)
				{
					input[myID]		= 0.0;
					input[emyID]	= 0.0;
				}else if(board[x][y] == bw)
				{
					input[myID]		= 1.0;
					input[emyID]	= 0.0;
				}else{
					input[myID]		= 0.0;
					input[emyID]	= 1.0;
				}
			}
		}

		Net.SetInput(input);
		Net.Forward();
		Net.GetOutput(output);

		for (int x = 0; x < BOARD_SIZE; ++x)
		{
			for (int y = 0; y < BOARD_SIZE; ++y)
			{
				int	copyBoard[BOARD_SIZE][BOARD_SIZE];
				memcpy(copyBoard, board, sizeof(copyBoard));

				if (check(&copyBoard, bw, x, y) > 0)
				{
					putCandidacy_t	tmp	=
					{
						MakeID(x, y),
						output[MakeID(x, y)],
					};

					array.push_back(tmp);
					total	+= tmp.ratio;
				}
			}
		}

		if (total == 0.0)
		{
			for (unsigned int i = 0; i < array.size(); ++i)
				array[i].ratio = 1.0 / array.size();
		}
		else {
			for (unsigned int i = 0; i < array.size(); ++i)
				array[i].ratio /= total;
		}
	}
	// ランダム配置.
	else
	{
		for (int x = 0; x < BOARD_SIZE; ++x)
		{
			for (int y = 0; y < BOARD_SIZE; ++y)
			{
				int	copyBoard[BOARD_SIZE][BOARD_SIZE];
				memcpy(copyBoard, board, sizeof(copyBoard));

				if (check(&copyBoard, bw, x, y) > 0)
				{
					putCandidacy_t	tmp	= {MakeID(x, y), 1.0};
					array.push_back(tmp);
					total	+= tmp.ratio;
				}
			}
		}

		for (unsigned int i = 0; i < array.size(); ++i)
			array[i].ratio /= total;
	}
	double	ratio	= dist(mt);

	for (auto a : array)
	{
		if (ratio < a.ratio)
		{
			if (learn)
			{
				log.id	= a.id;

				bwLog[bw].push_back(log);
			}

			return (a.id);
		}
		ratio -= a.ratio;
	}

	return (0);
}

/*----------------------------------------------------------------------
 * コンピュータの配置.
 *----------------------------------------------------------------------*/
static void CpuPut(int id, int bw)
{
	int	x, y, num;

	x	= IDtoX(id);
	y	= IDtoY(id);

	record[turn].x	= x;
	record[turn].y	= y;
	record[turn].bw	= bw;

	comCursorX	= x;
	comCursorY	= y;

	//調べてひっくり返す
	num	= check(&board, bw, x, y);

	//駒数の変更
	pieceNum[bw]	+= num + 1;
	pieceNum[SwapBW(bw)]	-= num;

	board[x][y]	= bw;
	++turn;

}

/*----------------------------------------------------------------------
 * 描画関数.
 *----------------------------------------------------------------------*/
static void Paint(HWND hWnd, HDC hdc)
{
	TCHAR	str[100];

	/* マスを書く */
	SelectObject(hdc, GetStockObject(BLACK_PEN));
	SelectObject(hdc, GetStockObject(NULL_BRUSH));

	Rectangle(hdc,
			  gridBase,
			  gridBase,
			  gridSize * BOARD_SIZE + gridBase + 1,
			  gridSize * BOARD_SIZE + gridBase + 1);

	for (int i = 1; i < BOARD_SIZE; i++)
	{
		MoveToEx(hdc,
				 gridBase + gridSize * i,
				 gridBase,
				 NULL);
		LineTo(  hdc,
				 gridBase + gridSize * i,
				 gridSize * BOARD_SIZE + gridBase + 1);

		MoveToEx(hdc,
				 gridBase,
				 gridBase + gridSize * i,
				 NULL);
		LineTo(  hdc,
				 gridSize * BOARD_SIZE + gridBase + 1,
				 gridBase + gridSize * i);
	}

	/* コンピュータの置いたマスを青く表示 */
	if (comCursorX >= 0 && comCursorX < BOARD_SIZE
	&&  comCursorY >= 0 && comCursorY < BOARD_SIZE)
	{
		SelectObject(hdc, GetStockObject(NULL_BRUSH));
		SelectObject(hdc, hPen2);

		Rectangle(hdc,
				  gridBase + gridSize * comCursorX,
				  gridBase + gridSize * comCursorY,
				  gridBase + gridSize * (comCursorX+1) + 1,
				  gridBase + gridSize * (comCursorY+1) + 1);
	}

	/* カーソルを書く*/
	if (manCursorX >= 0 && manCursorX < BOARD_SIZE
	&&  manCursorY >= 0 && manCursorY < BOARD_SIZE)
	{
		SelectObject(hdc, GetStockObject(NULL_BRUSH));
		SelectObject(hdc, hPen1);

		Rectangle(hdc,
				  gridBase + gridSize * manCursorX,
				  gridBase + gridSize * manCursorY,
				  gridBase + gridSize * (manCursorX+1) + 1,
				  gridBase + gridSize * (manCursorY+1) + 1);
	}

	SelectObject(hdc, GetStockObject(BLACK_PEN));

	for (int x = 0; x < BOARD_SIZE; ++x)
	{
		for (int y = 0; y < BOARD_SIZE; ++y)
		{
			switch (board[x][y])
			{
			  case BLACK:
				SelectObject(hdc, GetStockObject(BLACK_BRUSH));
				break;
			  case WHITE:
				SelectObject(hdc, GetStockObject(WHITE_BRUSH));
				break;

			  default:
				continue;
			}


			Ellipse(hdc,
					(int)(gridSize * (x+0.5) - gridSize *0.4) + gridBase,
					(int)(gridSize * (y+0.5) - gridSize *0.4) + gridBase,
					(int)(gridSize * (x+0.5) + gridSize *0.4) + gridBase,
					(int)(gridSize * (y+0.5) + gridSize *0.4) + gridBase);
		}
	}

	/* 黒と白の駒の数を表示 */
	wsprintf(str,
			 TEXT("黒 %d : 白 %d  "),
			 pieceNum[BLACK],
			 pieceNum[WHITE]);

	TextOut(hdc,
			gridBase*2 + gridSize * BOARD_SIZE,
			0,
			str,
			lstrlen(str));

	for (int i = 0; i < turn; ++i)
	{
		wsprintf(str,
				 TEXT("%s x %d y %d"),
				 record[i].bw ? TEXT("黒") : TEXT("白"),
				 record[i].x + 1,
				 record[i].y + 1);

		TextOut(hdc,
				gridBase*2 + gridSize * BOARD_SIZE + (100 * (i / 30)),
				gridBase*2 + 17 * (i % 30 + 1),
				str,
				lstrlen(str));
	}
}

/*----------------------------------------------------------------------
 * ゲーム初期化.
 *----------------------------------------------------------------------*/
static void gameInitialize(HWND hWnd)
{
	bool	sente;

	// 初期化
	format();

	if (learn)
	{
		std::random_device					rd;
		std::mt19937						mt(rd());
		std::uniform_int_distribution<>		dist(0, 2);

		sente = dist(mt) ? true : false;

		for (unsigned int i = 0; i < 2; ++i)
			bwLog[i].resize(0);

	}else {
		sente	= MessageBox(hWnd,
							 TEXT("先攻（黒）でいいですか？"),
							 TEXT("確認"),
							 MB_YESNO | MB_ICONINFORMATION) == IDYES
				? true : false;
	}

	// プレイヤーが先手
	if (sente)
	{
		manColor	= BLACK;
	}
	// プレイヤーが後攻
	else{
		manColor	= WHITE;
		int id	= Cpu(BLACK);

		int x	= IDtoX(id);
		int y	= IDtoY(id);

		check(&board, SwapBW(manColor), x, y);

		board[x][y] = BLACK;

		pieceNum[BLACK] = 4;
		pieceNum[WHITE] = 1;

		record[turn].x	= x + 1;
		record[turn].y	= y + 1;
		record[turn].bw	= BLACK;

		comCursorX	= x;
		comCursorY	= y;

		++turn;
	}
	comColor	= SwapBW(manColor);

}

/*----------------------------------------------------------------------
 * ゲームを終わらせる関数.
 *----------------------------------------------------------------------*/
static int gameEnd(HWND hWnd)
{
	TCHAR str[255];


	//点数の表示
	wsprintf(str
			 , TEXT("黒:%d 対 白:%d\n"),
			 pieceNum[BLACK],
			 pieceNum[WHITE]);

	// 勝敗の表示
	if (pieceNum[BLACK] > pieceNum[WHITE])
	{
		lstrcat(str, TEXT("黒の勝ちです"));
	}
	else if(pieceNum[WHITE] > pieceNum[BLACK])
	{
		lstrcat(str, TEXT("白の勝ちです"));
	}
	else{
		lstrcat(str, TEXT("引き分けです"));
	}

	MessageBox(hWnd,
			   str,
			   TEXT("確認"),
			   MB_OK | MB_ICONINFORMATION);

	return (0);
}

/*----------------------------------------------------------------------
 * 駒をおく関数.
 *----------------------------------------------------------------------*/
static void Put(HWND hWnd, int x, int y)
{
	int	num;

	/* Playerがおいたとき */
	if ((num = check(&board, manColor, x, y)) == 0)
	{
		MessageBox(hWnd,
				   TEXT("そこに置くことはできません"),
				   NULL,
				   MB_OK | MB_ICONEXCLAMATION);
		return;
	}

	pieceNum[manColor]	+= num + 1;
	pieceNum[comColor]	-= num;

	record[turn].x	= x;
	record[turn].y	= y;
	record[turn].bw	= manColor;

	board[x][y]		= manColor;
	++turn;

	// プレイヤーがおいて,駒の変化を表示.
	InvalidateRect(hWnd, NULL, FALSE);

	// 終了.
	if (turn >= (BOARD_SIZE*BOARD_SIZE-4))
	{
		gameEnd(hWnd);
		return;
	}

	if (!preCheck(comColor))
	{
		if (!preCheck(manColor))
		{
			//両方におく場所がないとき
			MessageBox(hWnd,
					   TEXT("両方とも置けるところがありません"),
					   TEXT("確認"),
					   MB_OK);

			gameEnd(hWnd);
			return;
		}

		//コンピュータにおく場所がないとき
		MessageBox(hWnd,
				   TEXT("相手には置けるところがありません"),
				   TEXT("確認"),
				   MB_OK);

		return;
	}

	do{
		int id = Cpu(comColor);

		CpuPut(id, comColor);

		InvalidateRect(hWnd, NULL, FALSE);

		// 終了確認.
		if (turn >= (BOARD_SIZE*BOARD_SIZE-4))
		{
			gameEnd(hWnd);
			return;
		}

		if (!preCheck(manColor))
		{
			if (!preCheck(comColor))
			{
				//両方ともおける場所がないとき
				MessageBox(hWnd,
						   TEXT("両方とも置けるところがありません"),
						   TEXT("確認"),
						   MB_OK);
				gameEnd(hWnd);
				return;
			}

			//プレイヤーにおく場所がないとき
			MessageBox(hWnd,
					   TEXT("あなたには置ける場所はありません"),
					   TEXT("確認"),
					   MB_OK | MB_ICONEXCLAMATION);
			continue;
		}
		break;
	} while(1);
}


/*----------------------------------------------------------------------
 * ウィンドウプロシージャ.
 *----------------------------------------------------------------------*/
static LRESULT CALLBACK WindowProc(HWND   hWnd,
								   UINT   uMsg,
								   WPARAM wParam,
								   LPARAM lParam)
{
	HDC				hdc;
	PAINTSTRUCT		ps;
	POINT			pt;
	int				x, y;

	switch (uMsg)
	{
		// 終了時.
	  case WM_DESTROY:
		DeleteObject(hPen1);
		DeleteObject(hPen2);
		PostQuitMessage(0);
		return (0);

		// 生成時.
	  case WM_CREATE:
		hPen1 = CreatePen(PS_DOT, 0, RGB(255,   0,   0));
		hPen2 = CreatePen(PS_DOT, 0, RGB(  0,   0, 255));

		gameInitialize(hWnd);

		return (0);

		// マウスの移動に併せてマスを表示.
	  case WM_MOUSEMOVE:
		x	= LOWORD(lParam);
		y	= HIWORD(lParam);

		manCursorX = (x - gridBase) / gridSize;
		manCursorY = (y - gridBase) / gridSize;

		if (manCursorX < 0)
		{
			manCursorX	= 0;
		}else if (manCursorX >= BOARD_SIZE)
		{
			manCursorX	= BOARD_SIZE-1;
		}

		if (manCursorY < 0)
		{
			manCursorY	= 0;
		}else if (manCursorY >= BOARD_SIZE)
		{
			manCursorY	= BOARD_SIZE-1;
		}

		InvalidateRect(hWnd, NULL, FALSE); //再描画
		return (0);

		// キー入力.
	  case WM_KEYDOWN:
		hdc = GetDC(hWnd);

		switch(wParam)
		{
			//マスの移動
		  case VK_UP:
			if (manCursorY > 0)
				--manCursorY;
			break;

		  case VK_DOWN:
			if (manCursorY < BOARD_SIZE)
				++manCursorY;
			break;

		  case VK_RIGHT:
			if (manCursorX < BOARD_SIZE)
				++manCursorX;
			break;

		  case VK_LEFT:
			if (manCursorX > 0)
				--manCursorX;
			break;

		  case VK_RETURN: //駒をおく
			pt.x	= gridBase + gridSize/2 + gridSize * manCursorX;
			pt.y	= gridBase + gridSize/2 + gridSize * manCursorY;

			ClientToScreen(hWnd, &pt);

			SetCursorPos(pt.x, pt.y);

			Put(hWnd, manCursorX, manCursorY);

			break;
		}

		pt.x	= gridBase + gridSize/2 + gridSize * manCursorX;
		pt.y	= gridBase + gridSize/2 + gridSize * manCursorY;

		ClientToScreen(hWnd, &pt);

		SetCursorPos(pt.x, pt.y);

		InvalidateRect(hWnd, NULL, FALSE);
		return (0);

		// 駒をおく.
	  case WM_LBUTTONUP:
		x = (LOWORD(lParam) - gridBase) / gridSize;
		y = (HIWORD(lParam) - gridBase) / gridSize;


		Put(hWnd, x, y);

		return (0);

		// 描画.
	  case WM_PAINT:
		hdc	= BeginPaint(hWnd, &ps);

		Paint(hWnd, hdc);

		EndPaint(hWnd, &ps);
		return (0);

		// メニュー.
	  case WM_COMMAND:
		switch(wParam)
		{
			//新規対戦
		  case ID_MENU_NEW:

			InvalidateRect(hWnd, NULL, TRUE);

			gameInitialize(hWnd);
			break;
		}
		return (0);
	}

	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}

/*======================================================================
 * エントリーポイント.
 *======================================================================*/
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmd,int nShow)
{
	WNDCLASS	wc;
	MSG			msg;

	// ニューラルネット読み込み
	{
		FILE	*pFile;
		char	buf[4];
		std::vector<char> data;

		pFile = fopen("othello.net", "rb");

		while (fread(buf, 1, sizeof(buf), pFile) > 0)
		{
			for (int i = 0; i < sizeof(buf); ++i)
				data.push_back(buf[i]);
		}

		fclose(pFile);

		Net.Load(data);
	}

	if (strcmp(lpCmd, "learn") == 0)
	{
		int		bw;

		learn	= true;

#if 0
		for (int i = 0; i < 10; ++i)
#endif
		{
			gameInitialize(0);
			bw	= manColor;

			// 終了.
			while (turn < (BOARD_SIZE*BOARD_SIZE-4))
			{
				CpuPut(Cpu(bw), bw);

				if (preCheck(SwapBW(bw)))
					bw	= SwapBW(bw);
			}

			if (pieceNum[BLACK] == pieceNum[WHITE])
				return (0);

			teacherData	log(BOARD_SIZE*BOARD_SIZE*2,
							BOARD_SIZE*BOARD_SIZE);

			log.Load("learning\\teacher.log");

			if (pieceNum[BLACK] > pieceNum[WHITE])
			{
				for (auto a : bwLog[BLACK])
					log.Add(a.input, a.id);
			}
			else if (pieceNum[WHITE] > pieceNum[BLACK])
			{
				for (auto a : bwLog[WHITE])
					log.Add(a.input, a.id);
			}

			log.Save("learning\\teacher.log");
		}
		return (0);
	}

	wc.style			= CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc		= WindowProc;
	wc.cbClsExtra		= 0;
	wc.cbWndExtra		= 0;
	wc.hInstance		= hInst;
	wc.hIcon			= LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground	= (HBRUSH)GetStockObject(WHITE_BRUSH);
	wc.lpszMenuName		= MAKEINTRESOURCE(IDR_MENU1);
	wc.lpszClassName	= APP_NAME;

	if (!RegisterClass(&wc))
		return (0);

	if ((g_hWnd = CreateWindow(APP_NAME, TEXT("OTHELLO"),
							   WS_VISIBLE | WS_OVERLAPPEDWINDOW,
							   CW_USEDEFAULT, CW_USEDEFAULT,
							   800, 640, NULL, NULL, hInst, NULL)) == NULL)
		return (0);

	while(GetMessage(&msg, NULL, 0, 0) > 0)
		DispatchMessage(&msg);

	return (msg.wParam);
}

