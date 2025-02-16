from abc import ABC, abstractmethod

class AbstractGame(ABC):
    """
    Abstrakt grensesnitt for et spillmiljø som kan brukes i MuZero.
    """

    @abstractmethod
    def reset(self):
        """
        Tilbakestiller spillet til starttilstanden.
        Returnerer den initiale observasjonen.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Utfører et steg i spillet basert på den gitte handlingen.
        
        Parametre:
            action (int eller annen datatype): Handlingen som skal utføres.
        
        Returnerer:
            observation: Den nye tilstanden etter handlingen.
            reward (float): Belønningen for det utførte steget.
            done (bool): Indikerer om episoden er ferdig.
            info (dict): Ekstra informasjon (kan være tomt).
        """
        pass

    @abstractmethod
    def legal_actions(self):
        """
        Returnerer en liste over alle gyldige handlinger i den nåværende tilstanden.
        Dette er nyttig for MuZero når man skal planlegge og evaluere mulige trekk.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Visualiserer den nåværende tilstanden i spillet.
        Denne metoden er valgfri og kan implementeres dersom visualisering er ønsket.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Rydder opp eventuelle ressurser (som vinduer eller filer) brukt av miljøet.
        """
        pass
