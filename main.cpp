#include "Eigen/Eigen"   // Aggiungiamo le librerie necessarie
#include "iostream"
#include "iomanip"

using namespace std;    // Specifichiamo i namespace
using namespace Eigen;

struct System {   // Si definisce uno struct
    MatrixXd A;
    VectorXd b;
};

VectorXd PALU(const MatrixXd& A, const VectorXd& b)  // Si definisce una funzione che attui la fattorizzazione PALU attraverso comandi
{                                                   // implementati nella libreria Eigen
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd QR(const MatrixXd& A, const VectorXd& b)  // Stessa cosa con la fattorizzazione QR
{
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}


double Err_Rel(const VectorXd& x, const VectorXd& x_esatta)  // Una funzione che calcola l'errore relativo (non necessaria ma creata per un
{                                                           // puro fattore estetico nel main
    double err =  (x - x_esatta).norm() / x_esatta.norm();
    return err;
}

int main() {
    vector<System> sistemi = {    // Definiamo un insieme di sistemi a partire dallo struct
        {
            MatrixXd(2, 2),
            VectorXd(2)
        },
        {
            MatrixXd(2,2),
            VectorXd(2)
        },
        {
            MatrixXd(2,2),
            VectorXd(2)
        }
    };

    sistemi[0].A << 5.547001962252291e-01, -3.770900990025203e-02,      //E da qui li esplicitiamo
        8.320502943378437e-01, -9.992887623566787e-01;

    sistemi[0].b << -5.169911863249772e-01, 1.672384680188350e-01;

    sistemi[1].A << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;

    sistemi[1].b << -6.394645785530173e-04, 4.259549612877223e-04;

    sistemi[2].A << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    sistemi[2].b << -6.400391328043042e-10, 4.266924591433963e-10;

    VectorXd sol(2);  // Definiamo il vettore soluzione esatta
    sol << -1.0e+00, -1.0e+00;


    for (const System& s : sistemi) {  // Creiamo un ciclo for che scorra sui valori delle coppie A e b dello struct
        cout << "Matrice A:\n" << s.A << endl;
        cout << "Vettore b:\n" << s.b << endl;
        VectorXd x_p = PALU(s.A, s.b); // Dove per ognuno si calcola la fattorizzazione PALU
        VectorXd x_q = QR(s.A,s.b); // quella QR
        cout << "Soluzione con la decomposizione PALU:\n" << x_p << endl;
        cout << "Errore relativo con la decomposizione PALU:\n" << Err_Rel(x_p,sol)<< endl;  //e gli errori relativi tramite la funzione sopra definita
        cout << "Soluzione con la decomposizione QR:\n" << x_q << endl;
        cout << "Errore relativo con la decomposizione QR:\n" << Err_Rel(x_q,sol) <<"\n"<< endl;

    };

    return 0;
}
